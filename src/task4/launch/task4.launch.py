#!/usr/bin/env python3
"""
Task 4 Launch File for Autonomous Agrar-Roboter (Knicklenker)

This launch file starts all required nodes for:
1. Marker detection (Marker_Detector_CV)
2. Global marker mapping (Global_Marker_Map_Node)
3. Coverage planner (Fields2Cover Integration)
4. Nav2 configuration

Author: FRE2026 Team
"""

from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # Define paths
    task4_pkg = FindPackageShare("task4")
    laser_scan_merger_pkg = FindPackageShare("laser_scan_merger")
    slam_toolbox_pkg = FindPackageShare("slam_toolbox")

    # Define launch arguments
    robotname = LaunchConfiguration("robotname")
    use_sim_time = LaunchConfiguration("use_sim_time", default="false")

    # Declare launch arguments
    arg_robotname = DeclareLaunchArgument(
        name="robotname",
        default_value="",
        description="Robot name for laser scan merger configuration",
    )

    arg_use_sim_time = DeclareLaunchArgument(
        name="use_sim_time",
        default_value="false",
        description="Use simulation time if true",
    )

    # ==================== Marker Detector Node ====================
    # marker_detector_node = Node(
    #     package="task4",
    #     executable="marker_detector",
    #     name="marker_detector",
    #     output="screen",
    #     parameters=[
    #         PathJoinSubstitution([task4_pkg, "config", "marker_detector.yaml"])
    #     ],
    #     remappings=[
    #         ("camera/image_raw", "/sensors/camera/image_raw"),
    #         ("camera/camera_info", "/sensors/camera/camera_info"),
    #         ("detected_marker", "/detected_marker"),
    #     ],
    #     arguments=["--ros-args", "--log-level", "info"],
    # )

    # ==================== Global Marker Map Node ====================
    # global_marker_map_node = Node(
    #     package="task4",
    #     executable="global_marker_map",
    #     name="global_marker_map",
    #     output="screen",
    #     parameters=[
    #         PathJoinSubstitution([task4_pkg, "config", "global_marker_map.yaml"])
    #     ],
    #     remappings=[
    #         ("detected_marker", "/detected_marker"),
    #         ("detected_markers", "/detected_markers"),
    #         ("active_markers", "/active_markers"),
    #     ],
    #     arguments=["--ros-args", "--log-level", "info"],
    # )

    # ==================== Coverage Planner Node ====================
    coverage_planner_node = Node(
        package="task4",
        executable="coverage_planner",
        name="coverage_planner",
        output="screen",
        parameters=[
            PathJoinSubstitution([task4_pkg, "config", "coverage_planner.yaml"])
        ],
        remappings=[
            ("global_plan", "/plan"),
            ("local_plan", "/local_plan"),
            ("field_boundary", "/field_boundary"),
            ("target", "/target"),
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    # ==================== Build Launch Description ====================
    # Note: For isolated testing of coverage_planner, only the coverage_planner_node is included.
    # The Nav2 stack and other nodes are commented out. Uncomment as needed for full integration.
    ld = LaunchDescription([
        arg_robotname,
        arg_use_sim_time,
        # marker_detector_node,
        # global_marker_map_node,
        coverage_planner_node
    ])

    return ld