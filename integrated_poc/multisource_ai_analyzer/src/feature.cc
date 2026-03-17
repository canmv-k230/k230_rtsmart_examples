#include <dirent.h>
#include <vector>
#include "feature.h"
#include <filesystem>

// Alias for filesystem namespace
namespace fs = std::filesystem;

/**
 * @brief Feature extractor constructor
 *
 * Initializes the Feature model by loading the kmodel,
 * setting input/output shapes, and preparing ai2d input tensor.
 *
 * @param kmodel_file  Path to the feature extraction kmodel
 * @param image_size   Input image size in CHW format
 * @param debug_mode   Debug level
 */
Feature::Feature(const char *kmodel_file,
                 FrameCHWSize image_size,
                 int debug_mode)
    : AIBase(kmodel_file, "Feature", debug_mode)
{
    // Override model name for timing and debug output
    model_name_ = "Feature";

    // Store original image size
    image_size_ = image_size;

    // Extract model input size (C, H, W) from kmodel input shape
    input_size_ = {
        input_shapes_[0][1],
        input_shapes_[0][2],
        input_shapes_[0][3]
    };

    // Feature vector length from model output shape
    feature_num_ = output_shapes_[0][1];

    // Cache ai2d output tensor (model input tensor)
    ai2d_out_tensor_ = get_input_tensor(0);
}

/**
 * @brief Feature extractor destructor
 */
Feature::~Feature()
{
}

/**
 * @brief Pre-process input image for feature extraction
 *
 * Crops the region of interest from the input image using the given bounding box,
 * then resizes it to match the model input size using ai2d.
 *
 * @param input_tensor Input runtime tensor containing the source image
 * @param box          Bounding box specifying the region of interest
 */
void Feature::pre_process(runtime_tensor& input_tensor, cv::Rect& box)
{
    // Measure preprocessing time
    ScopedTiming st(model_name_ + " pre_process", debug_mode_);

    // Configure ai2d for crop and resize
    Utils::crop_resize_set(
        image_size_,
        input_size_,
        (int)box.x,
        (int)box.y,
        (int)box.width,
        (int)box.height,
        ai2d_builder_
    );

    // Execute ai2d preprocessing
    ai2d_builder_->invoke(input_tensor, ai2d_out_tensor_)
        .expect("error occurred in ai2d running");
}

/**
 * @brief Run feature extraction inference
 *
 * Executes the model and retrieves output tensors.
 */
void Feature::inference()
{
    this->run();
    this->get_output();
}

/**
 * @brief Get extracted feature vector
 *
 * Copies the feature embedding from model output buffer
 * into a user-provided vector.
 *
 * @param feature Output vector to store the feature embedding
 */
void Feature::get_feature(vector<float> &feature)
{
    // Measure feature extraction copy time
    ScopedTiming st(model_name_ + " get_feature", debug_mode_);

    // Resize output vector to feature dimension
    feature.resize(feature_num_);

    // Copy feature data from model output
    std::memcpy(
        feature.data(),
        p_outputs_[0],
        feature_num_ * sizeof(float)
    );
}
