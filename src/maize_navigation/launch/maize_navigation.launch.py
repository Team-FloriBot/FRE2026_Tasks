from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = Path(get_package_share_directory("maize_navigation"))
    params_file = pkg_share / "config" / "params.yaml"

    return LaunchDescription([
        DeclareLaunchArgument(
            "simulate_tracked_objects",
            default_value="false",
            description="Publish simulated tracked objects from the object tracker.",
        ),
        DeclareLaunchArgument(
            "tracker_tf_lookup_offset_sec",
            default_value="0.10",
            description="Subtract this offset from detection stamps before TF lookup.",
        ),
        DeclareLaunchArgument(
            "tracker_tf_timeout_sec",
            default_value="0.20",
            description="How long the object tracker waits for TF data.",
        ),
        Node(
            package="maize_navigation",
            executable="navigator",
            name="maize_navigator",
            output="screen",
            parameters=[str(params_file)],
        ),
        Node(
            package="fre2026_detection_client",
            executable="object_tracker",
            name="object_tracker",
            output="screen",
            parameters=[{
                "simulation_enabled": LaunchConfiguration("simulate_tracked_objects"),
                "tf_lookup_offset_sec": LaunchConfiguration("tracker_tf_lookup_offset_sec"),
                "tf_timeout_sec": LaunchConfiguration("tracker_tf_timeout_sec"),
            }],
        ),
    ])
