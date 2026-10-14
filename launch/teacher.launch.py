import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('autonoms_car')
    config_file = os.path.join(pkg_dir, 'config', 'params.yaml')

    teacher_node = Node(
        package='autonoms_car',
        executable='rule_based_teacher',
        name='rule_based_teacher_node',
        output='screen',
        parameters=[config_file]
    )

    return LaunchDescription([teacher_node])
