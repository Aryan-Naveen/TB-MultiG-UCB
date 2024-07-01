# The big distributed seeking node. This reduces the number of topic subscription needed.
import os
import sys
import traceback
import time

import numpy as np

from functools import partial
from collections import deque

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray,Bool,String
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile

from ucb_interfaces.action import UCBEpisode
from geometry_msgs.msg import Pose


from ucb_interfaces.msg import UCBTrajectory

import numpy as np
from geometry_msgs.msg import Pose
from ucb_interfaces.msg import UCBTrajectory


def xy2traj(xy_points):
    trajectory = UCBTrajectory()
    for point in xy_points:
        trajectory.poses.append(xy2Pose(point))
    
    return trajectory

def xy2Pose(xy):
    pose = Pose()
    pose.position.x = xy[0]
    pose.position.y = 0.0
    pose.position.z = xy[1]
    return pose

class test(Node):
    def __init__(self,xlims=[-np.inf,np.inf],ylims = [-np.inf,np.inf], csize=0.2):
        super().__init__(node_name = 'UCB')
        self.waypoints = np.array([[0, 0], [-csize, 0], [-2*csize, 0], [-3*csize, 0], [-4*csize, 0], [-5*csize, 0], [-6*csize, 0], [-6*csize, -1*csize], [-6*csize, -2*csize], [-6*csize, -3*csize]])

        self.traj = xy2traj(self.waypoints)

        qos = QoSProfile(depth=10)

        self._agent_publisher = self.create_publisher(UCBTrajectory, '/MobileSensor5/UCB_trajectory', qos)
    
    def send_goal(self):
        self.traj.episode_duration = 30.0
        self._agent_publisher.publish(self.traj)

    

def main(args=None):
    rclpy.init(args=args)

    test_node = test()
    test_node.send_goal()
    rclpy.shutdown()

if __name__ == '__main__':
    main()