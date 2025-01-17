from functools import partial

import rclpy
from rclpy.qos import QoSProfile
from rcl_interfaces.srv import GetParameters

from std_msgs.msg import Float32MultiArray,Float32

from .pose import get_pose_type_and_topic,toxy,toyaw

from collections import deque

from ucb_interfaces.msg import UCBAgentPackage
import copy
import numpy as np

class robot_listener:
	''' Robot location listener+data container.'''
	def __init__(self,controller_node,robot_namespace,pose_type_string="",max_record_len=10):
		"""
			pose_type_string is one in ["turtlesimPose", "Pose", "Odom", "optitrack"]
		"""
		self.robot_name=robot_namespace

		self.controller_node = controller_node
		controller_node.get_logger().info('initializing {} listener'.format(robot_namespace))
		
		
		self.pose_type,self.rpose_topic=get_pose_type_and_topic(pose_type_string,robot_namespace)
		
		self.robot_pose_stack = deque(maxlen=10)
		
		qos = QoSProfile(depth=10)

		controller_node.create_subscription(self.pose_type, self.rpose_topic,self.robot_pose_callback_,qos)

		controller_node.create_subscription(UCBAgentPackage, '/{}/UCB_rewards'.format(robot_namespace),self.reward_callback,qos)

		self.last_reward = None


	def get_latest_loc(self):
		if len(self.robot_pose_stack)>0:
			return toxy(self.robot_pose_stack[-1])
		else:
			return None

	def get_latest_yaw(self):
		if len(self.robot_pose_stack)>0:
			return toyaw(self.robot_pose_stack[-1])
		else:
			return None

	def get_latest_episode_rewards(self):
		reward = copy.deepcopy(self.last_reward)
		self.last_reward = None
		return reward

	def robot_pose_callback_(self,data):
		self.robot_pose_stack.append(data)

	def latest_episode_concluded(self):
		return self.last_reward is not None

	def reward_callback(self,data):
		self.last_reward = data.rewards