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


from ucb_interfaces.msg import UCBTrajectory, UCBAgentPackage, UCBPackageNode

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


        qos = QoSProfile(depth=10)

        self.pubs = {}

        self.namespaces = ['MobileSensor5', 'MobileSensor4', 'MobileSensor3', 'MobileSensor2', 'MobileSensor1']
        self.rewards = {}

        nodes = [int(100*i + j) for i in range(6) for j in range(6)]
        for namespace in self.namespaces:
            self.pubs[namespace] = self.create_publisher(UCBAgentPackage, '/{}/UCB_rewards'.format(namespace), qos)
            self.rewards[namespace] = UCBAgentPackage()
            for i, node in enumerate(nodes):
                reward = UCBPackageNode()
                reward.id = node
                reward.interval = [5*i, 5*i + 4]
                reward.mean = 100.0
                self.rewards[namespace].rewards.append(reward)

    def send_rewards(self):
        for namespace in self.namespaces:
            self.pubs[namespace].publish(self.rewards[namespace])
    

def main(args=None):
    rclpy.init(args=args)

    test_node = test()
    test_node.send_rewards()
    rclpy.shutdown()

if __name__ == '__main__':
    main()