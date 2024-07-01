from functools import partial

import rclpy
from rclpy.qos import QoSProfile
from rcl_interfaces.srv import GetParameters

from std_msgs.msg import Float32MultiArray,Float32

from .pose import xy2traj

from collections import deque

from ucb_interfaces.msg import UCBTrajectory


class robot_publisher:
    ''' Robot UCB Publishing container.'''
    def __init__(self,controller_node,robot_namespace, max_record_len=10):
        """
            pose_type_string is one in ["turtlesimPose", "Pose", "Odom", "optitrack"]
        """
        self.robot_name=robot_namespace

        self.controller_node = controller_node
        controller_node.get_logger().info('initializing {} publisher'.format(robot_namespace))

        qos = QoSProfile(depth=max_record_len)

        self.trajectory_publisher = self.controller_node.create_publisher(UCBTrajectory, '/{}/UCB_trajectory'.format(robot_namespace), qos)		

    def publish_trajectory(self, trajectory, episode_duration):
        traj_msg = xy2traj(trajectory)
        traj_msg.episode_duration = episode_duration
        self.trajectory_publisher.publish(traj_msg)
