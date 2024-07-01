from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    ucb_config_pkg = get_package_share_directory('ucb_interfaces')
    param_file_path = os.path.join(ucb_config_pkg, 'config', 'ucb_params.yaml')

    ld = LaunchDescription()
    action  = Node(
            package='ucb',
            executable='ucb',
            name='MAB_node',
            parameters=[param_file_path],
            output='screen',
        )
    ld.add_action(action)
    return ld
