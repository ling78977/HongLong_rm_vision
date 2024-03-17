// Copyright 2021 RoboMaster-OSS
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

#ifndef RMOSS_PROJECTILE_MOTION__GAF_PROJECTILE_SOLVER_HPP_
#define RMOSS_PROJECTILE_MOTION__GAF_PROJECTILE_SOLVER_HPP_

#include <string>
#include <memory>

#include "rmoss_projectile_motion/projectile_solver_interface.hpp"
#include "rmoss_projectile_motion/iterative_projectile_tool.hpp"

namespace rmoss_projectile_motion
{

// 考虑重力和空气阻力（gaf:Gravity and Air Frication）的弹道重力修正工具.
class GafProjectileSolver : public ProjectileSolverInterface
{
public:
  GafProjectileSolver(double initial_vel, double friction_coeff);

  void set_initial_vel(double vel) {initial_vel_ = vel;}
  void set_friction_coeff(const double & friction_coeff) {friction_coeff_ = friction_coeff;}
  std::shared_ptr<IterativeProjectileTool> get_iterative_tool() {return iterative_tool_;}
  bool solve(double target_x, double target_h, double & angle) override;
  std::string error_message() override;

private:
  std::shared_ptr<IterativeProjectileTool> iterative_tool_;
  // 子弹射速
  double initial_vel_;
  // 空气阻力系数
  double friction_coeff_;
};

}  // namespace rmoss_projectile_motion

#endif  // RMOSS_PROJECTILE_MOTION__GAF_PROJECTILE_SOLVER_HPP_
