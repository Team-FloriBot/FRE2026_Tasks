from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():

    default_config = PathJoinSubstitution([
        FindPackageShare("path_tracking_controller"),
        "config",
        "params.yaml"
    ])

    config_arg = DeclareLaunchArgument(
        "config",
        default_value=default_config,
        description="Parameter file for Pure Pursuit Controller"
    )

    pure_pursuit_node = Node(
        package="path_tracking_controller",
        executable="pure_pursuit_node",
        name="pure_pursuit_node",
        parameters=[LaunchConfiguration("config")],
        output="screen",
        arguments=["--ros-args", "--log-level", "info"]
    )

    return LaunchDescription([
        config_arg,
        pure_pursuit_node
    ])