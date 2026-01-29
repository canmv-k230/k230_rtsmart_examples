#ifndef _FEATURE_H
#define _FEATURE_H

#include <vector>
#include "ai_utils.h"
#include "ai_base.h"

using std::vector;

/**
 * @brief Face recognition class based on MobileFaceNet
 *
 * This class encapsulates the complete processing pipeline for face feature
 * extraction on each frame, including preprocessing, inference, and
 * postprocessing to obtain the final feature vector.
 */
class Feature : public AIBase
{
public:
    /**
     * @brief Constructor of Feature
     * @param kmodel_file Path to the kmodel file
     * @param isp_shape   Input image size in CHW format
     * @param debug_mode  Debug mode (0: no log, 1: timing only, 2: full logs)
     */
    Feature(char *kmodel_file, FrameCHWSize isp_shape, int debug_mode);

    /**
     * @brief Destructor of Feature
     * @return None
     */
    ~Feature();

    /**
     * @brief Preprocess input image and ROI before inference
     * @param input_tensor Input runtime tensor (source image)
     * @param box          Face bounding box region
     * @return None
     */
    void pre_process(runtime_tensor& input_tensor, cv::Rect& box);

    /**
     * @brief Run kmodel inference
     * @return None
     */
    void inference();

    /**
     * @brief Get extracted face feature vector
     * @param feature Output feature vector
     * @return None
     */
    void get_feature(vector<float> &feature);

private:
    std::unique_ptr<ai2d_builder> ai2d_builder_; // ai2d builder for preprocessing
    runtime_tensor ai2d_out_tensor_;             // ai2d output tensor
    
    FrameCHWSize image_size_;                    // Original input image size
    FrameCHWSize input_size_;                    // Model input size
    int feature_num_;                            // Length of extracted face feature
    float obj_thresh_;                           // Face recognition threshold
};
#endif
