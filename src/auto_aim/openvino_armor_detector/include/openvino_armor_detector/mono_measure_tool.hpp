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

#ifndef OPENVINO_ARMOR_DETECTOR__MONO_MEASURE_TOOL_HPP_
#define OPENVINO_ARMOR_DETECTOR__MONO_MEASURE_TOOL_HPP_

#include <geometry_msgs/msg/point.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include <openvino_armor_detector/types.hpp>

namespace rm_auto_aim {

class MonoMeasureTool {
public:
  MonoMeasureTool() = default;

  explicit MonoMeasureTool(std::vector<double> camera_intrinsic,
                           std::vector<double> camera_distortion);

  /**
   * @brief Set the camera intrinsic parameter
   *
   * @param camera_intrinsic camera intrinsic in 3x3 matrix flat in line stretch
   * @param camera_distortion camera distortion parameter in plumb_bob
   * distortion model
   * @return true
   * @return false
   */
  bool setCameraInfo(std::vector<double> camera_intrinsic,
                     std::vector<double> camera_distortion);
  /**
   * @brief Solve Perspective-n-Point problem in camera
   * 3d点坐标求解（use solve pnp）
   * @param points2d a list of points in image frame
   * @param points3d a list of points correspondend to points2d
   * @param position output position of the origin point of 3d coordinate system
   * @return true
   * @return false
   */
  bool solvePnP(const ArmorObject &obj, cv::Mat &rvec, cv::Mat &tvec);
  /**
   * @brief 逆投影，已知深度，2d->3d点求解
   *
   * @param p 图像上点坐标
   * @param distance 已知的真实距离
   * @return cv::Point3f 对应的真实3d点坐标
   */
  cv::Point3f unproject(cv::Point2f p, double distance);
  /**
   * @brief 视角求解
   *
   * @param p 图像上点坐标
   * @param pitch 视角pitch
   * @param yaw 视角yaw
   */
  void calc_view_angle(cv::Point2f p, float &pitch, float &yaw);

  /**
   * @brief 装甲板目标位姿求解
   *
   * @param obj 装甲板目标
   * @param position 返回的坐标
   * @param rvec 相对旋转向量
   * @return true
   * @return false
   */
  bool calc_armor_target(const ArmorObject &obj, cv::Point3f &position,
                         cv::Mat &rvec);

  float calcDistanceToCenter(const ArmorObject &obj);

private:

  std::vector<cv::Point3f> small_armor_points_;
  std::vector<cv::Point3f> large_armor_points_;
  // 相机参数
  cv::Mat camera_intrinsic_;  // 相机内参3*3
  cv::Mat camera_distortion_; // 相机畸变参数1*5

  // Unit: mm
  static constexpr float SMALL_ARMOR_WIDTH = 135;
  static constexpr float SMALL_ARMOR_HEIGHT = 55;
  static constexpr float LARGE_ARMOR_WIDTH = 225;
  static constexpr float LARGE_ARMOR_HEIGHT = 55;

};

} // namespace rm_auto_aim

#endif // OPENVINO_ARMOR_DETECTOR__MONO_MEASURE_TOOL_HPP_
