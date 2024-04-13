// Copyright 2023 Yunlong Feng
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPENVINO_ARMOR_DETECTOR__OPENVINO_DETECTOR_HPP_
#define OPENVINO_ARMOR_DETECTOR__OPENVINO_DETECTOR_HPP_

#include <Eigen/Dense>

#include <cmath>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <openvino_armor_detector/mono_measure_tool.hpp>

#include <openvino_armor_detector/types.hpp>

namespace rm_auto_aim {

struct GridAndStride {
  int grid0;
  int grid1;
  int stride;
};

// struct SimLight {
//   SimLight() = default;
//   explicit SimLight(const cv::Point2f &pt1, const cv::Point2f &pt2) {
//     length = std::sqrt(std::pow(pt1.x - pt2.x, 2) + std::pow(pt1.y - pt2.y, 2));
//     center = cv::Point2f((pt1.x + pt2.x) / 2, (pt1.y + pt2.y) / 2);
//   }
//   double length;
//   cv::Point2f center;
// };

class OpenVINODetector {

public:
  // bool isBigArmor(const ArmorObject &obj);
  /**
   * @brief Construct a new OpenVINO Detector object
   *
   * @param model_path IR/ONNX file path
   * @param device_name Target device (CPU, GPU, AUTO)
   * @param conf_threshold Confidence threshold for output filtering
   * @param top_k Topk parameter
   * @param nms_threshold NMS threshold
   * @param auto_init If initializing detector inplace
   */
  explicit OpenVINODetector(const std::filesystem::path &model_path,
                            const std::string &device_name,
                            float conf_threshold = 0.25, int top_k = 128,
                            float nms_threshold = 0.3, bool auto_init = false);

  /**
   * @brief Initialize detector
   *
   */
  void init();

  


  bool detect(cv::Mat &rgb_img, std::vector<ArmorObject> &objs);

private:
  std::string model_path_;
  std::string device_name_;
  float conf_threshold_;
  int top_k_;
  float nms_threshold_;
  std::vector<int> strides_;
  std::vector<GridAndStride> grid_strides_;

  std::unique_ptr<ov::Core> ov_core_;
  std::unique_ptr<ov::CompiledModel> compiled_model_; // 可执行网络

  ov::InferRequest infer_request_; // 推理请求

  Eigen::Matrix<float, 3, 3> transform_matrix_;
};
} // namespace rm_auto_aim

#endif // OPENVINO_ARMOR_DETECTOR__OPENVINO_DETECTOR_HPP_
