from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    package_share_directory=get_package_share_directory("rm_vision_bringup")
    rviz_config_file=os.path.join(package_share_directory,'config','visual_debug.rviz')
    rviz_node=Node(
        package="rviz2",
        executable='rviz2',
        name='rviz2',
        arguments=['-d',rviz_config_file]
    )
    return LaunchDescription([
        rviz_node
    ])