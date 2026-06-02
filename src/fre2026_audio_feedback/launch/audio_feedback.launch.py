from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("fre2026_audio_feedback"),
        "config",
        "audio_feedback.yaml",
    )

    return LaunchDescription([
        Node(
            package="fre2026_audio_feedback",
            executable="audio_feedback_node",
            name="audio_feedback_node",
            output="screen",
            parameters=[config],
        )
    ])
