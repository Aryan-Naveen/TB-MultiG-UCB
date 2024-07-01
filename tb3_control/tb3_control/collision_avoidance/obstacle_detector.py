import os
import sys

tools_root = os.path.join(".."+os.path.dirname(__file__))
print(tools_root)
sys.path.insert(0, os.path.abspath(tools_root))


import numpy as np

from rcl_interfaces.srv import GetParameters

from ros2_utils.robot_listener import robot_listener
from ros2_utils.param_service_client import param_service_client
from ros2_utils.misc import get_sensor_names,get_source_names
from collision_avoidance import regions

# The radius of a Turtlebot Burger. Useful in collision avoidance.
BURGER_RADIUS = 0.110

class obstacle_detector:
	def __init__(self,mc_node):
		self.obs_names = get_sensor_names(mc_node)
		self.ol = [robot_listener(mc_node,name,mc_node.pose_type_string) for name in self.obs_names if (not name == mc_node.robot_namespace) and (not name=='Source0')]

	def get_free_spaces(self):
		
		SAFE_RADIUS = 3*BURGER_RADIUS
		# SAFE_RADIUS = 5*BURGER_RADIUS
		# SAFE_RADIUS = 6*BURGER_RADIUS

		obs = [(l.get_latest_loc(),SAFE_RADIUS) for l in self.ol if not l.get_latest_loc() is None]

		return [regions.CircleExterior(origin,radius) for (origin,radius) in obs]

class boundary_detector:
	def __init__(self,controller_node,xlims,ylims):
		self.xlims = xlims
		self.ylims = ylims
		# self.xlims = (-1e5,1e5)
		# self.ylims = (-1e5,1e5)
		
		# # Get boundary services.
		# self.param_names = ['xlims','ylims']
		# self.param_service = '/MISSION_CONTROL/boundary/get_parameters'

		# self.boundary_client = param_service_client(controller_node,self.param_names,self.param_service)


	def get_free_spaces(self):
		# result = self.boundary_client.get_params()
		# if len(result)>0:
		# 		[self.xlims,self.ylims] = result

		return [regions.Rect2D(self.xlims,self.ylims)]


