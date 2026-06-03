from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="floribot_scan_mux",
            executable="front_scan_mux",
            name="front_scan_mux",
            output="screen",
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare("floribot_scan_mux"),
                    "config",
                    "front_scan_mux.yaml",
                ])
            ],
        ),
    ])
