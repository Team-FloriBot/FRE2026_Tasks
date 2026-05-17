from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = Path(get_package_share_directory("maize_navigation"))
    params_file = pkg_share / "config" / "params.yaml"

    return LaunchDescription([
        Node(
            package="maize_navigation",
            executable="navigator",
            name="maize_navigator",
            output="screen",
            parameters=[str(params_file)],
        )
    ])
