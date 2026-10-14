import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('autonoms_car')
    config_file = os.path.join(pkg_dir, 'config', 'params.yaml')

    mock_lidar_node = Node(
        package='autonoms_car',
        executable='mock_lidar',
        name='mock_lidar_node',
        output='screen',
        parameters=[config_file]
    )

    ai_controller_node = Node(
        package='autonoms_car',
        executable='ai_controller',
        name='ai_controller_node',
        output='screen',
        parameters=[config_file]
    )

    return LaunchDescription([
        mock_lidar_node,
        ai_controller_node
    ])
