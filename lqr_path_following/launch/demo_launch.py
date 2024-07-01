from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    ucb_config_pkg = get_package_share_directory('ucb_interfaces')
    param_file_path = os.path.join(ucb_config_pkg, 'config', 'ucb_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'param_file',
            default_value=param_file_path,
            description='Path to the parameter file'
        ),
        Node(
            package='tb3_control',
            executable='tb3_controller',
            namespace='MobileSensor1',
            name='agent_node',
            parameters=[[LaunchConfiguration('param_file')]],
            output='screen',
        ),
    ])
