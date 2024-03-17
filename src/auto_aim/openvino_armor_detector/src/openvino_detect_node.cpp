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

#include "openvino_armor_detector/openvino_detect_node.hpp"

#include <cv_bridge/cv_bridge.h>
#include <fmt/format.h>
#include <rmw/qos_profiles.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <rmoss_util/url_resolver.hpp>

namespace rm_auto_aim {
OpenVINODetectNode::OpenVINODetectNode(rclcpp::NodeOptions options)
    : Node("openvino_detect_node", options.use_intra_process_comms(true)) {
  RCLCPP_INFO(this->get_logger(), "Initializing detect node");

  RCLCPP_INFO(this->get_logger(), "Initializing OpenVINO");
  detector_ = nullptr;
  this->init_detector();
  if (!detector_) {
    RCLCPP_ERROR(this->get_logger(), "Failed to initialize OpenVINO");
    return;
  }
  rcl_interfaces::msg::ParameterDescriptor param_desc;
  param_desc.description = "0-RED, 1-BLUE";
  param_desc.integer_range.resize(1);
  param_desc.integer_range[0].from_value = 0;
  param_desc.integer_range[0].to_value = 1;
  detect_color_ = this->declare_parameter("detect_color", 0, param_desc);

  auto use_sensor_data_qos =
      this->declare_parameter("use_sensor_data_qos", false);

  camera_name_ = this->declare_parameter("detector.camera_name", "camera");
  transport_type_ =
      this->declare_parameter("detector.subscribe_compressed", false)
          ? "compressed"
          : "raw";
  RCLCPP_INFO(this->get_logger(), "camera_name: %s, transport_type: %s",
              camera_name_.c_str(), transport_type_.c_str());

  // Debug mode handler
  RCLCPP_INFO(this->get_logger(), "Setup debug_mode handler");
  debug_mode_ = this->declare_parameter("debug_mode", false);
  if (debug_mode_) {
    this->create_debug_publishers();
  }
  // Regiter debug mode param handler
  debug_param_sub_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
  debug_cb_handle_ = debug_param_sub_->add_parameter_callback(
      "debug_mode", [this](const rclcpp::Parameter &p) {
        this->debug_mode_ = p.as_bool();
        this->debug_mode_ ? this->create_debug_publishers()
                          : this->destroy_debug_publishers();
      });

  RCLCPP_INFO(this->get_logger(), "Setup ROS subs pubs");
  // Armors publisher
  armors_pub_ = this->create_publisher<auto_aim_interfaces::msg::Armors>(
      "detector/armors", rclcpp::SensorDataQoS());

  // Visualization Marker
  armor_marker_.ns = "armors";
  armor_marker_.action = visualization_msgs::msg::Marker::ADD;
  armor_marker_.type = visualization_msgs::msg::Marker::CUBE;
  armor_marker_.scale.x = 0.05;
  armor_marker_.scale.z = 0.125;
  armor_marker_.color.a = 1.0;
  armor_marker_.color.g = 0.5;
  armor_marker_.color.b = 1.0;
  armor_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);

  text_marker_.ns = "classification";
  text_marker_.action = visualization_msgs::msg::Marker::ADD;
  text_marker_.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  text_marker_.scale.z = 0.1;
  text_marker_.color.a = 1.0;
  text_marker_.color.r = 1.0;
  text_marker_.color.g = 1.0;
  text_marker_.color.b = 1.0;
  text_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);
  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "detector/marker", 10);

  // Camera handler
  cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      "/camera_info",
      use_sensor_data_qos ? rclcpp::SensorDataQoS() : rclcpp::QoS(1),
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info) {
        this->cam_info_ =
            std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
        this->measure_tool_ = std::make_unique<MonoMeasureTool>(
            std::vector<double>(this->cam_info_->k.begin(),
                                this->cam_info_->k.end()),
            this->cam_info_->d);

        RCLCPP_INFO(this->get_logger(),
                    fmt::format("Camera intrinsic: {} \ncamera distortion: {}",
                                fmt::join(this->cam_info_->k, " "),
                                fmt::join(this->cam_info_->d, " "))
                        .c_str());

        // Release subscription
        this->cam_info_sub_.reset();
      });
  auto qos = use_sensor_data_qos ? rmw_qos_profile_sensor_data
                                 : rmw_qos_profile_default;
  //设置队列长度为1，保证实时性
  qos.depth = 1;
  img_sub_ = std::make_shared<image_transport::Subscriber>(
      image_transport::create_subscription(
          this, "/image_raw",
          std::bind(&OpenVINODetectNode::img_callback, this,
                    std::placeholders::_1),
          transport_type_, qos));
  RCLCPP_INFO(this->get_logger(), "Subscribing to %s",
              img_sub_->getTopic().c_str());

  RCLCPP_INFO(this->get_logger(), "Initializing finished.");
}

void OpenVINODetectNode::init_detector() {
  auto model_path = this->declare_parameter("detector.model_path", "");
  auto device_type = this->declare_parameter("detector.device_type", "AUTO");
  float conf_threshold =
      this->declare_parameter("detector.confidence_threshold", 0.7);
  int top_k = this->declare_parameter("detector.top_k", 128);
  float nms_threshold = this->declare_parameter("detector.nms_threshold", 0.3);

  if (model_path.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Require model_path but got empty.");
    return;
  }
  auto resolved_path = rmoss_util::URLResolver::get_resolved_path(model_path);
  if (resolved_path.empty()) {
    RCLCPP_ERROR(
        this->get_logger(),
        "Model file path format error. Should be package://<package>/<path> or "
        "file:///<path>");
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Model path: %s", resolved_path.c_str());

  // Create detector
  detector_ = std::make_unique<OpenVINODetector>(
      resolved_path, device_type, conf_threshold, top_k, nms_threshold);
  // Set detect callback
  detector_->set_callback(std::bind(
      &OpenVINODetectNode::openvino_detect_callback, this,
      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
  // init detector
  detector_->init();
}

void OpenVINODetectNode::img_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
  // limits request size
  while (detect_requests_.size() > 1) {
    detect_requests_.front().get();
    detect_requests_.pop();
  }

  auto timestamp = rclcpp::Time(msg->header.stamp);
  // auto timestamp = rclcpp::Time(this->get_clock()->now());
  frame_id_ = msg->header.frame_id;
  auto img = cv_bridge::toCvCopy(msg, "rgb8")->image;

  // push image to detector
  detect_requests_.push(
      std::move(detector_->push_input(img, timestamp.nanoseconds())));
}

void OpenVINODetectNode::openvino_detect_callback(
    const std::vector<ArmorObject> &objs, int64_t timestamp_nanosec,
    const cv::Mat &src_img) {
  if (measure_tool_ == nullptr) {
    RCLCPP_WARN(this->get_logger(), "No camera_info recieve yet.");
    return;
  }

  detect_color_ = this->get_parameter("detect_color").as_int();

  auto timestamp = rclcpp::Time(timestamp_nanosec);

  // Used to draw debug info
  cv::Mat debug_img;
  if (debug_mode_) {
    debug_img = src_img.clone();
  }
  if (measure_tool_ != nullptr) {

    armors_msg_.header.frame_id = frame_id_;
    armors_msg_.header.stamp = timestamp;
    armor_marker_.header.frame_id = frame_id_;
    armor_marker_.header.stamp = timestamp;
    text_marker_.header.frame_id = frame_id_;
    text_marker_.header.stamp = timestamp;
    armors_msg_.armors.clear();
    marker_array_.markers.clear();
    armor_marker_.id = 0;
    text_marker_.id = 0;

    auto_aim_interfaces::msg::Armor armor_msg;

    for (auto &obj : objs) {
      if (detect_color_ == 0 && obj.color != ArmorColor::RED) {
        continue;
      } else if (detect_color_ == 1 && obj.color != ArmorColor::BLUE) {
        continue;
      }
      auto tempobj = obj;
      tempobj.is_big = detector_->isBigArmor(obj);

      cv::Mat rvec, tvec;
      bool success = measure_tool_->solvePnP(tempobj, rvec, tvec);
      if (success) {
        // Fill basic info
        armor_msg.type = tempobj.is_big ? "large" : "small";
        armor_msg.number = kArmorNames[static_cast<int>(obj.number)];

        // Fill pose
        armor_msg.pose.position.x = tvec.at<double>(0);
        armor_msg.pose.position.y = tvec.at<double>(1);
        armor_msg.pose.position.z = tvec.at<double>(2);

        cv::Mat rot_mat;
        cv::Rodrigues(rvec, rot_mat);
        tf2::Matrix3x3 tf_rot_mat(
            rot_mat.at<double>(0, 0), rot_mat.at<double>(0, 1),
            rot_mat.at<double>(0, 2), rot_mat.at<double>(1, 0),
            rot_mat.at<double>(1, 1), rot_mat.at<double>(1, 2),
            rot_mat.at<double>(2, 0), rot_mat.at<double>(2, 1),
            rot_mat.at<double>(2, 2));
        tf2::Quaternion tf_quaternion;
        tf_rot_mat.getRotation(tf_quaternion);
        armor_msg.pose.orientation = tf2::toMsg(tf_quaternion);
        // Fill the distance to image center
        armor_msg.distance_to_image_center =
            measure_tool_->calcDistanceToCenter(obj);

        // Fill the markers
        armor_marker_.id++;
        armor_marker_.scale.y = tempobj.is_big ? 0.23 : 0.135;
        armor_marker_.pose = armor_msg.pose;
        text_marker_.id++;
        text_marker_.pose.position = armor_msg.pose.position;
        text_marker_.pose.position.y -= 0.1;
        std::string armor_key =
            fmt::format("{}:{}", tempobj.is_big ? "big" : "small",
                        static_cast<int>(obj.number));
        text_marker_.text = armor_key;

        armors_msg_.armors.emplace_back(armor_msg);
        marker_array_.markers.emplace_back(armor_marker_);
        marker_array_.markers.emplace_back(text_marker_);

      } else {
        RCLCPP_WARN(this->get_logger(), "PnP failed!");
      }
    }
    // Publishing marker
    publishMarkers();

    //// Publishing detected armors
    armors_pub_->publish(std::move(armors_msg_));

    auto final_time = this->get_clock()->now();
    auto latency = (final_time.seconds() - timestamp.seconds()) * 1000;
    RCLCPP_INFO_STREAM(this->get_logger(), "Latency: " << latency << "ms");

    // Publish debug info
    if (debug_mode_) {
      if (debug_img.empty()) {
        // Avoid debug_mode change in processing
        return;
      }
      for (auto &obj : objs) {
        for (size_t i = 0; i < 4; ++i) {
          cv::line(debug_img, obj.pts[i], obj.pts[(i + 1) % 4],
                   cv::Scalar(0, 255, 0), 2);
          cv::circle(debug_img, obj.pts[i], 5, cv::Scalar(255, 0, 255), 2);
        }

        std::string armor_color;
        switch (obj.color) {
        case ArmorColor::BLUE:
          armor_color = "blue";
          break;
        case ArmorColor::RED:
          armor_color = "red";
          break;
        case ArmorColor::NONE:
          armor_color = "none";
          break;
        case ArmorColor::PURPLE:
          armor_color = "purple";
          break;
        default:
          armor_color = "UNKOWN";
          break;
        }
        auto tempobj = obj;
        tempobj.is_big = detector_->isBigArmor(obj);

        std::string armor_key = fmt::format("{} {} {}", armor_color,
                                            tempobj.is_big ? "big" : "small",
                                            static_cast<int>(obj.number));
        cv::putText(debug_img, armor_key, cv::Point2i(obj.pts[0]),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
      }
      cv::circle(debug_img,
                 cv::Point2i(cam_info_->width / 2., cam_info_->height / 2.), 5,
                 cv::Scalar(255, 255, 255), 2);
      auto end = this->get_clock()->now();
      auto duration = end.seconds() - timestamp.seconds();
      std::string latency = fmt::format("Latency: {:.3f}ms", duration * 1000);
      cv::putText(debug_img, latency, cv::Point2i(10, 30),
                  cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
      debug_img_pub_.publish(
          cv_bridge::CvImage(armors_msg_.header, "rgb8", debug_img)
              .toImageMsg());
    }
  }
}

void OpenVINODetectNode::create_debug_publishers() {
  debug_img_pub_ =
      image_transport::create_publisher(this, "detector/debug_img");
}

void OpenVINODetectNode::destroy_debug_publishers() {
  debug_img_pub_.shutdown();
}

void OpenVINODetectNode::publishMarkers() {
  using Marker = visualization_msgs::msg::Marker;
  armor_marker_.action =
      armors_msg_.armors.empty() ? Marker::DELETE : Marker::ADD;
  marker_array_.markers.emplace_back(std::move(armor_marker_));
  marker_pub_->publish(marker_array_);
}

} // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable
// when its library is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::OpenVINODetectNode)
