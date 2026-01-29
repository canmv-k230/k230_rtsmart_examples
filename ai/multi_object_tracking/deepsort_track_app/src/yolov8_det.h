#ifndef _YOLOV8_DET_H
#define _YOLOV8_DET_H

#include "ai_utils.h"
#include "ai_base.h"

/**
 * @brief Structure representing a single YOLO detection bounding box
 */
typedef struct YOLOBbox{
    cv::Rect box;        ///< Bounding box coordinates (x, y, width, height)
    float confidence;    ///< Detection confidence score
    int index;           ///< Class index of the detected object
} YOLOBbox;

/**
 * @brief YOLOv8 multi-object detection class
 *
 * This class encapsulates the complete detection pipeline for each frame,
 * including preprocessing, model inference, and postprocessing to obtain
 * final detection results.
 */
class YOLOv8Det : public AIBase
{
public:

    /**
     * @brief Constructor for YOLOv8Det
     *
     * Loads the kmodel and initializes model input/output tensors as well as
     * detection thresholds.
     *
     * @param kmodel_file Path to the kmodel file
     * @param score_thres Confidence score threshold for detections
     * @param nms_thres   Non-Maximum Suppression (NMS) threshold
     * @param image_size  Input image size (CHW format)
     * @param debug_mode  Debug mode:
     *                    0 - no debug output
     *                    1 - timing information only
     *                    2 - verbose debug information
     */
    YOLOv8Det(char *kmodel_file, float score_thres, float nms_thres, FrameCHWSize image_size, int debug_mode);
    
    /**
     * @brief Destructor for YOLOv8Det
     */
    ~YOLOv8Det();

    /**
     * @brief Preprocess input image data
     *
     * This function prepares the input tensor for inference, typically
     * including resizing, padding, normalization, and layout conversion.
     *
     * @param input_tensor Runtime tensor to be used as model input
     */
    void pre_process(runtime_tensor &input_tensor);

    /**
     * @brief Run kmodel inference
     *
     * Executes the forward pass of the YOLOv8 model.
     */
    void inference();

    /**
     * @brief Postprocess inference results
     *
     * Decodes raw model outputs, applies confidence thresholding and NMS,
     * and maps detection results back to the original image coordinates.
     *
     * @param results Vector to store final detection results
     */
    void post_process(vector<YOLOBbox> &results);

private:

    /**
     * @brief Perform Non-Maximum Suppression (NMS) on detection boxes
     *
     * @param bboxes         Input bounding boxes
     * @param confThreshold Confidence threshold
     * @param nmsThreshold  NMS IoU threshold
     * @param indices       Output indices of selected boxes after NMS
     */
    void yolov8_nms(std::vector<YOLOBbox> &bboxes,
                    float confThreshold,
                    float nmsThreshold,
                    std::vector<int> &indices);
    
    /**
     * @brief Calculate Intersection over Union (IoU) between two rectangles
     *
     * @param rect1 First rectangle
     * @param rect2 Second rectangle
     * @return IoU value in range [0, 1]
     */
    float yolov8_iou_calculate(cv::Rect &rect1, cv::Rect &rect2);

    std::unique_ptr<ai2d_builder> ai2d_builder_; ///< ai2d preprocessing builder
    runtime_tensor ai2d_in_tensor_;              ///< ai2d input tensor
    runtime_tensor ai2d_out_tensor_;             ///< ai2d output tensor
    FrameCHWSize image_size_;                    ///< Original image size
    FrameCHWSize input_size_;                    ///< Model input image size

    /**
     * @brief List of class labels (COCO dataset)
     */
    std::vector<std::string> labels{
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    int label_num_ = 80;                          ///< Number of object classes

    std::vector<cv::Scalar> colors;               ///< Colors for drawing bounding boxes
    float conf_thres_;                            ///< Confidence threshold
    float nms_thres_;                             ///< NMS IoU threshold
    int box_num_;                                 ///< Total number of detected boxes
    int max_box_num_ = 50;                        ///< Maximum number of output boxes
    int box_feature_len_;                         ///< Feature length per detection box

    int max_ = 0;                                 ///< Auxiliary variable (usage depends on implementation)
};

#endif
