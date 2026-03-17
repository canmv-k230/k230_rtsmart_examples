#include "yolov8_det.h"

/**
 * @brief Constructor for YOLOv8Det
 *
 * Initializes the YOLOv8 detection model, loads the kmodel, configures
 * preprocessing (ai2d), detection thresholds, and internal buffers.
 */
YOLOv8Det::YOLOv8Det(const char *kmodel_file,
                     float score_thres,
                     float nms_thres,
                     FrameCHWSize image_size,
                     int debug_mode)
: AIBase(kmodel_file,"YOLOv8Det", debug_mode)
{
    model_name_ = "YOLOv8Det";                     ///< Model name identifier
    conf_thres_ = score_thres;                    ///< Confidence threshold
    nms_thres_  = nms_thres;                      ///< NMS IoU threshold
    image_size_ = image_size;                     ///< Original image size

    // Model input size (CHW) obtained from kmodel input shape
    input_size_ = {
        input_shapes_[0][1],
        input_shapes_[0][2],
        input_shapes_[0][3]
    };

    // Generate visualization colors for each class
    colors = getColorsForClasses(label_num_);

    max_box_num_ = 100;                            ///< Maximum number of output boxes

    // Total number of prediction boxes from all detection heads
    box_num_ =
        ((input_size_.width  / 8)  * (input_size_.height / 8)) +
        ((input_size_.width  / 16) * (input_size_.height / 16)) +
        ((input_size_.width  / 32) * (input_size_.height / 32));

    debug_mode_ = debug_mode;                      ///< Debug mode configuration

    // ai2d output tensor maps directly to model input tensor
    ai2d_out_tensor_ = get_input_tensor(0);

    // Feature length per detection box: [x, y, w, h] + class scores
    box_feature_len_ = label_num_ + 4;

    // Configure ai2d preprocessing: resize with padding on one side
    Utils::padding_resize_one_side_set(
        image_size_,
        input_size_,
        ai2d_builder_,
        cv::Scalar(114, 114, 114)                  ///< Padding color
    );
}

/**
 * @brief Destructor for YOLOv8Det
 */
YOLOv8Det::~YOLOv8Det()
{
}

/**
 * @brief Preprocess input frame before inference
 *
 * Applies ai2d preprocessing such as resize, padding, and layout conversion.
 *
 * @param input_tensor Input runtime tensor (raw image data)
 */
void YOLOv8Det::pre_process(runtime_tensor &input_tensor)
{
    ScopedTiming st(model_name_ + " pre_process video", debug_mode_);
    ai2d_builder_->invoke(input_tensor, ai2d_out_tensor_)
        .expect("error occurred in ai2d running");
}

/**
 * @brief Run YOLOv8 inference
 *
 * Executes the forward pass and retrieves model outputs.
 */
void YOLOv8Det::inference()
{
    this->run();
    this->get_output();
}

/**
 * @brief Postprocess YOLOv8 inference results
 *
 * Decodes raw model output, applies confidence thresholding,
 * rescales boxes back to original image coordinates, and
 * performs Non-Maximum Suppression (NMS).
 *
 * @param results Output vector of detected bounding boxes
 */
void YOLOv8Det::post_process(vector<YOLOBbox> &results)
{
    ScopedTiming st(model_name_ + " post_process", debug_mode_);

    // Calculate resize ratio (keeping aspect ratio)
    float ratiow = (float)input_size_.width  / image_size_.width;
    float ratioh = (float)input_size_.height / image_size_.height;
    float ratio  = ratiow < ratioh ? ratiow : ratioh;

    // Allocate buffer for transposed output
    float *output_det = new float[box_num_ * box_feature_len_];

    // Transpose output tensor to [box_num, feature_len]
    transpose_block_fast(
        p_outputs_[0],
        output_det,
        box_num_,
        box_feature_len_
    );

    // Iterate through all predicted boxes
    for (int i = 0; i < box_num_; i++) {

        float* vec = output_det + i * box_feature_len_;
        float box[4] = { vec[0], vec[1], vec[2], vec[3] };

        // Pointer to class confidence scores
        float* class_scores = vec + 4;

        // Find class with maximum confidence
        float* max_class_score_ptr =
            std::max_element(class_scores, class_scores + label_num_);

        float score = *max_class_score_ptr;
        int max_class_index = max_class_score_ptr - class_scores;

        // Filter by confidence threshold and target class (person = 0)
        if (score > conf_thres_ && max_class_index == 0) {

            YOLOBbox bbox;

            // Rescale box coordinates back to original image space
            float x_ = box[0] / ratio;
            float y_ = box[1] / ratio;
            float w_ = box[2] / ratio;
            float h_ = box[3] / ratio;

            int x = int(MAX(x_ - 0.5 * w_, 0));
            int y = int(MAX(y_ - 0.5 * h_, 0));
            int w = int(w_);
            int h = int(h_);

            // Clamp box within image boundaries
            if (x + w > image_size_.width)
                w = image_size_.width - x;

            if (y + h > image_size_.height)
                h = image_size_.height - y;

            // Avoid negative or invalid dimensions
            w = std::max(w, 0);
            h = std::max(h, 0);
            if (w <= 0 || h <= 0)
                continue;

            bbox.box        = cv::Rect(x, y, w, h);
            bbox.confidence = score;
            bbox.index      = max_class_index;

            results.push_back(bbox);
        }
    }

    // 执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
    std::vector<int> nms_result;
    yolov8_nms(results, conf_thres_, nms_thres_, nms_result);

    delete[] output_det;
}

/**
 * @brief Perform Non-Maximum Suppression (NMS)
 *
 * Suppresses overlapping bounding boxes with lower confidence scores.
 *
 * @param bboxes         Input/output bounding boxes
 * @param confThreshold Confidence threshold
 * @param nmsThreshold  IoU threshold for suppression
 * @param indices       Output indices of retained boxes
 */
void YOLOv8Det::yolov8_nms(std::vector<YOLOBbox> &bboxes,
                           float confThreshold,
                           float nmsThreshold,
                           std::vector<int> &indices)
{
    // Sort boxes by confidence in descending order
    std::sort(
        bboxes.begin(),
        bboxes.end(),
        [](YOLOBbox &a, YOLOBbox &b) {
            return a.confidence > b.confidence;
        }
    );

    int updated_size = bboxes.size();

    for (int i = 0; i < updated_size; i++) {

        if (bboxes[i].confidence < confThreshold)
            continue;

        indices.push_back(i);

        // Mark overlapping boxes instead of erasing to reduce memory movement
        for (int j = i + 1; j < updated_size; j++) {
            float iou =
                yolov8_iou_calculate(bboxes[i].box, bboxes[j].box);
            if (iou > nmsThreshold) {
                bboxes[j].confidence = -1; // Mark as suppressed
            }
        }
    }

    // Remove suppressed boxes
    bboxes.erase(
        std::remove_if(
            bboxes.begin(),
            bboxes.end(),
            [](YOLOBbox &b) { return b.confidence < 0; }
        ),
        bboxes.end()
    );
}

/**
 * @brief Calculate Intersection over Union (IoU)
 *
 * Computes the IoU metric between two rectangles.
 *
 * @param rect1 First rectangle
 * @param rect2 Second rectangle
 * @return IoU value in range [0, 1]
 */
float YOLOv8Det::yolov8_iou_calculate(cv::Rect &rect1, cv::Rect &rect2)
{
    int xx1, yy1, xx2, yy2;

    xx1 = std::max(rect1.x, rect2.x);
    yy1 = std::max(rect1.y, rect2.y);
    xx2 = std::min(rect1.x + rect1.width  - 1,
                   rect2.x + rect2.width - 1);
    yy2 = std::min(rect1.y + rect1.height - 1,
                   rect2.y + rect2.height - 1);

    int insection_width  = std::max(0, xx2 - xx1 + 1);
    int insection_height = std::max(0, yy2 - yy1 + 1);

    float insection_area =
        float(insection_width) * insection_height;

    float union_area =
        float(rect1.width * rect1.height +
              rect2.width * rect2.height -
              insection_area);

    float iou = insection_area / union_area;

    return iou;
}
