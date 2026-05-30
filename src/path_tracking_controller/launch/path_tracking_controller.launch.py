from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Konfigurationsdatei
    config_file = LaunchConfiguration('config')
    
    return LaunchDescription([
        # Launch-Argument für Konfigurationsdatei
        DeclareLaunchArgument(
            'config',
            default_value=['$(find-pkg-share path_tracking_controller)/config/params.yaml'],
            description='Pfad zur Konfigurationsdatei für den Pure Pursuit Controller'),
            
        # Pure Pursuit Controller Node
        Node(
            package='path_tracking_controller',
            executable='pure_pursuit_node',
            name='pure_pursuit_node',
            parameters=[config_file],
            output='screen',
            arguments=['--ros-args', '--log-level', 'info']
        ),
    ])
