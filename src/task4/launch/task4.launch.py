import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Pfad zur Parameter-Datei dynamisch auflösen
    config_dir = os.path.join(
        get_package_share_directory('task4'),
        'config',
        'coverage_planner.yaml'
    )

    coverage_planner_node = Node(
        package='task4',
        executable='coverage_planner',
        name='coverage_planner',
        output='screen',
        parameters=[config_dir],
    )

    return LaunchDescription([
        coverage_planner_node
    ])