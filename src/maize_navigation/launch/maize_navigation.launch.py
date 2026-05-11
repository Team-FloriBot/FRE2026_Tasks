from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('maize_navigation')
    param_file = os.path.join(pkg_dir, 'config', 'params.yaml')

    return LaunchDescription([
        Node(
            package='maize_navigation',
            executable='navigator',
            name='maize_navigator',
            output='screen',
            parameters=[param_file]
        )
    ])
