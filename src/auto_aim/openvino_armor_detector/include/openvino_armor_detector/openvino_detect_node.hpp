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

#ifndef OPENVINO_ARMOR_DETECTOR__OPENVINO_DETECT_NODE_HPP_
#define OPENVINO_ARMOR_DETECTOR__OPENVINO_DETECT_NODE_HPP_

#include <future>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <auto_aim_interfaces/msg/armors.hpp>
#include <openvino_armor_detector/mono_measure_tool.hpp>
#include <openvino_armor_detector/openvino_detector.hpp>
#include <openvino_armor_detector/types.hpp>

namespace rm_auto_aim {

class OpenVINODetectNode : public rclcpp::Node {
public:
  OpenVINODetectNode(rclcpp::NodeOptions options);

private:
  void initDetector();

  void imgCallback(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg);

  std::string getText(ArmorObject &obj);

  void fillArmorMsg(auto_aim_interfaces::msg::Armor &armor_msg,ArmorObject &obj);

  void publishArmorsMsg(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg, std::vector<ArmorObject> &objs);

  void publishDebugImage(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg,
                         std::vector<ArmorObject> &objs);

  // Debug functions
  void createDebugPublishers();

  void destroyDebugPublishers();

  void publishMarkers();

private:
  std::string camera_name_;
  std::string transport_type_;
  std::string frame_id_;
  // OpenVINO Detector
  int detect_color_; // 0: red, 1: blue
  std::unique_ptr<OpenVINODetector> detector_;
  std::queue<std::future<bool>> detect_requests_;
  // Camera info
  std::shared_ptr<sensor_msgs::msg::CameraInfo> cam_info_;
  std::unique_ptr<MonoMeasureTool> measure_tool_;

  // Visualization marker publisher
  visualization_msgs::msg::Marker armor_marker_;
  visualization_msgs::msg::Marker text_marker_;
  visualization_msgs::msg::MarkerArray marker_array_;
  auto_aim_interfaces::msg::Armors armors_msg_;

  // ROS
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      marker_pub_;
  rclcpp::Publisher<auto_aim_interfaces::msg::Armors>::SharedPtr armors_pub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
  std::shared_ptr<image_transport::Subscriber> img_sub_;

  // Debug publishers
  bool debug_mode_{false};
  std::shared_ptr<rclcpp::ParameterEventHandler> debug_param_sub_;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> debug_cb_handle_;
  image_transport::Publisher debug_img_pub_;
};

} // namespace rm_auto_aim

#endif // OPENVINO_ARMOR_DETECTOR__OPENVINO_DETECT_NODE_HPP_
