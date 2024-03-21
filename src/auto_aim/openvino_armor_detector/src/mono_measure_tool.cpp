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

#include <openvino_armor_detector/mono_measure_tool.hpp>

namespace rm_auto_aim {

MonoMeasureTool::MonoMeasureTool(std::vector<double> camera_intrinsic,
                                 std::vector<double> camera_distortion) {
  setCameraInfo(camera_intrinsic, camera_distortion);
  // Unit: m
  constexpr double small_half_y = SMALL_ARMOR_WIDTH / 2.0 / 1000.0;
  constexpr double small_half_z = SMALL_ARMOR_HEIGHT / 2.0 / 1000.0;
  constexpr double large_half_y = LARGE_ARMOR_WIDTH / 2.0 / 1000.0;
  constexpr double large_half_z = LARGE_ARMOR_HEIGHT / 2.0 / 1000.0;

  // Start from bottom left in clockwise order
  // Model coordinate: x forward, y left, z up
  small_armor_points_.emplace_back(cv::Point3f(0, small_half_y, -small_half_z));
  small_armor_points_.emplace_back(cv::Point3f(0, small_half_y, small_half_z));
  small_armor_points_.emplace_back(cv::Point3f(0, -small_half_y, small_half_z));
  small_armor_points_.emplace_back(
      cv::Point3f(0, -small_half_y, -small_half_z));

  large_armor_points_.emplace_back(cv::Point3f(0, large_half_y, -large_half_z));
  large_armor_points_.emplace_back(cv::Point3f(0, large_half_y, large_half_z));
  large_armor_points_.emplace_back(cv::Point3f(0, -large_half_y, large_half_z));
  large_armor_points_.emplace_back(
      cv::Point3f(0, -large_half_y, -large_half_z));

  filted_small_armor_points_.emplace_back(cv::Point3f(0, small_half_y, 0));
  filted_small_armor_points_.emplace_back(cv::Point3f(0, 0, -small_half_z));
  filted_small_armor_points_.emplace_back(cv::Point3f(0, -small_half_y, 0));
  filted_small_armor_points_.emplace_back(cv::Point3f(0, 0, small_half_z));

  filted_large_armor_points_.emplace_back(cv::Point3f(0, large_half_y, 0));
  filted_large_armor_points_.emplace_back(cv::Point3f(0, 0, -large_half_z));
  filted_large_armor_points_.emplace_back(cv::Point3f(0, -large_half_y, 0));
  filted_large_armor_points_.emplace_back(cv::Point3f(0, 0, large_half_z));
}

bool MonoMeasureTool::setCameraInfo(std::vector<double> camera_intrinsic,
                                    std::vector<double> camera_distortion) {
  if (camera_intrinsic.size() != 9) {
    // the size of camera intrinsic must be 9 (equal 3*3)
    return false;
  }
  // init camera_intrinsic and camera_distortion
  cv::Mat camera_intrinsic_mat(camera_intrinsic, true);
  camera_intrinsic_mat = camera_intrinsic_mat.reshape(0, 3);
  camera_intrinsic_ = camera_intrinsic_mat.clone();

  cv::Mat camera_distortion_mat(camera_distortion, true);
  camera_distortion_mat = camera_distortion_mat.reshape(0, 1);
  camera_distortion_ = camera_distortion_mat.clone();
  return true;
}

bool MonoMeasureTool::solvePnP(const ArmorObject &obj, cv::Mat &rvec,
                               cv::Mat &tvec) {
  std::vector<cv::Point2f> image_armor_points;
  // Fill in image points
  image_armor_points.emplace_back(obj.pts[1]);
  image_armor_points.emplace_back(obj.pts[0]);
  image_armor_points.emplace_back(obj.pts[3]);
  image_armor_points.emplace_back(obj.pts[2]);

  auto object_points = obj.is_big ? large_armor_points_ : small_armor_points_;
  // auto object_points = small_armor_points_;

  return cv::solvePnP(object_points, image_armor_points, camera_intrinsic_,
                      camera_distortion_, rvec, tvec, false, cv::SOLVEPNP_IPPE);
}

bool MonoMeasureTool::solveFiltedPnP(const ArmorObject &obj, cv::Mat &rvec,
                                     cv::Mat &tvec) {
  std::vector<cv::Point2f> image_armor_points;
  image_armor_points.emplace_back((obj.pts[0] + obj.pts[1]) / 2);
  image_armor_points.emplace_back((obj.pts[1] + obj.pts[2]) / 2);
  image_armor_points.emplace_back((obj.pts[2] + obj.pts[3]) / 2);
  image_armor_points.emplace_back((obj.pts[3] + obj.pts[0]) / 2);

  auto object_points =
      obj.is_big ? filted_large_armor_points_ : filted_small_armor_points_;

  return cv::solvePnP(object_points, image_armor_points, camera_intrinsic_,
                      camera_distortion_, rvec, tvec, false, cv::SOLVEPNP_IPPE);
}

// refer to :http://www.cnblogs.com/singlex/p/pose_estimation_1_1.html
// 根据输入的参数将图像坐标转换到相机坐标中
// 输入为图像上的点坐标
// double distance 物距
// 输出3d点坐标的单位与distance（物距）的单位保持一致
cv::Point3f MonoMeasureTool::unproject(cv::Point2f p, double distance) {
  auto fx = camera_intrinsic_.ptr<double>(0)[0];
  auto u0 = camera_intrinsic_.ptr<double>(0)[2];
  auto fy = camera_intrinsic_.ptr<double>(1)[1];
  auto v0 = camera_intrinsic_.ptr<double>(1)[2];

  double zc = distance;
  double xc = (p.x - u0) * distance / fx;
  double yc = (p.y - v0) * distance / fy;
  return cv::Point3f(xc, yc, zc);
}

// 获取image任意点的视角，pitch，yaw（相对相机坐标系）。
// 与相机坐标系保持一致。
void MonoMeasureTool::calc_view_angle(cv::Point2f p, float &pitch, float &yaw) {
  auto fx = camera_intrinsic_.ptr<double>(0)[0];
  auto u0 = camera_intrinsic_.ptr<double>(0)[2];
  auto fy = camera_intrinsic_.ptr<double>(1)[1];
  auto v0 = camera_intrinsic_.ptr<double>(1)[2];

  pitch = atan2((p.y - v0), fy);
  yaw = atan2((p.x - u0), fx);
}

// bool MonoMeasureTool::calc_armor_target(const ArmorObject &obj,
//                                         cv::Point3f &position, cv::Mat &rvec)
//                                         {
//   if (is_big_armor(obj)) {
//     return solve_pnp(obj.pts, big_armor_3d_points, position, rvec,
//                      cv::SOLVEPNP_IPPE);
//   } else {
//     return solve_pnp(obj.pts, small_armor_3d_points, position, rvec,
//                      cv::SOLVEPNP_IPPE);
//   }
// }

float MonoMeasureTool::calcDistanceToCenter(const ArmorObject &obj) {
  cv::Point2f img_center(this->camera_intrinsic_.at<double>(0, 2),
                         this->camera_intrinsic_.at<double>(1, 2));
  cv::Point2f armor_center;
  armor_center.x =
      (obj.pts[0].x + obj.pts[1].x + obj.pts[2].x + obj.pts[3].x) / 4.;
  armor_center.y =
      (obj.pts[0].y + obj.pts[1].y + obj.pts[2].y + obj.pts[3].y) / 4.;
  auto dis_vec = img_center - armor_center;
  return sqrt(dis_vec.dot(dis_vec));
}

}  // namespace rm_auto_aim
