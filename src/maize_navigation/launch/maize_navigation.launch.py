from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("maize_navigation")
    params_file = os.path.join(pkg_share, "config", "params.yaml")

    navigator = Node(
        package="maize_navigation",
        executable="navigator",
        name="maize_navigator",
        output="screen",
        parameters=[params_file],
    )

    return LaunchDescription([
        navigator,
    ])
