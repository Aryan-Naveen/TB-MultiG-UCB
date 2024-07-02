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
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.qos import QoSProfile



tools_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(tools_root))

# General dependencies
from ros2_utils.robot_listener import robot_listener
from ros2_utils.pose import prompt_pose_type_string
from ros2_utils.misc import get_sensor_names

# Waypoint planning dependencies
from motion_control import WaypointPlanning

from motion_control.WaypointTracking import BURGER_MAX_LIN_VEL, BURGER_MAX_ANG_VEL

from util_func import analytic_dLdp, joint_F_single, top_n_mean

# Motion control dependencies
from ros2_utils.pose import bounded_change_update, turtlebot_twist, stop_twist

from motion_control.WaypointTracking import PID_controller, PID

from collision_avoidance.obstacle_detector import obstacle_detector, boundary_detector
from collision_avoidance.regions import RegionsIntersection

# UCB algorithm dependencies
from ucb_interfaces.action import UCBEpisode
from ucb_interfaces.msg import UCBTrajectory, UCBPackageNode, UCBAgentPackage

from ros2_utils.ucb import dict2UCBPackage
from ros2_utils.pose import ros_poses2numpy


from rclpy.executors import MultiThreadedExecutor
# Coefficient names
COEF_NAMES = ['C1','C0','k','b']




class agent_node(Node):
    def __init__(self,robot_namespace):
        super().__init__(node_name = 'agent_node', namespace = robot_namespace)
        self.declare_parameters(
            namespace='UCB_params',
            parameters=[
                ('neighborhood_namespaces', rclpy.Parameter.Type.STRING_ARRAY),
                ('pose_type_string', rclpy.Parameter.Type.STRING),
                ('xlims', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('ylims', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('cell_size', rclpy.Parameter.Type.DOUBLE)
            ]
        )


        self.robot_namespace = robot_namespace        

        self.declare_parameter('neighborhood_namespaces', ["MobileSensor{}".format(i) for i in range(1,6)])
        self.neighborhood_namespaces = self.get_parameter('neighborhood_namespaces').get_parameter_value().string_array_value


        try:
            assert(robot_namespace in self.neighborhood_namespaces)
        except:
            print(f"Robot {robot_namespace} not in active robot list!!!")
            return

        self.declare_parameter('pose_type_string', 'optitrack')
        self.pose_type_string = self.get_parameter('pose_type_string').get_parameter_value().string_value


        self.robot_listeners = {namespace:robot_listener(self,namespace,self.pose_type_string, )\
                         for namespace in self.neighborhood_namespaces}



        qos = QoSProfile(depth=10)

        """
        UCB Initialization
        """
        self.node_list = []
        self.reward_list = []
        self._ucb_subscriber = self.create_subscription(UCBTrajectory, '/{}/UCB_trajectory'.format(robot_namespace), self.ucb_callback, qos)
        self._ucb_publisher = self.create_publisher(UCBAgentPackage, '/{}/UCB_rewards'.format(robot_namespace), qos)

        """
        Timer initialization
        """

        self.reward_sleep_time = 1e-1
        
        self.reward_collection_timer = self.create_timer(self.reward_sleep_time,self.reward_callback)

        self.motion_sleep_time = 5e-1

        self.motion_timer = self.create_timer(self.motion_sleep_time,self.motion_callback)

        """ 
        Waypoint planning initializations 
        """
            
        # Temporary hard-coded waypoints used in devel.	
        # self.waypoints = np.array([[-0.0, -0.0], [-0.0, -0.6], [-0.6, -0.6]])
        self.waypoints = np.array([])

        pos_PIDs = PID(Kp=10, Ki=0.0, Kd=0.0, max_output=BURGER_MAX_LIN_VEL)
        theta_PIDs = PID(Kp=1, Ki=0.0, Kd=0.00, max_output=BURGER_MAX_ANG_VEL)
        self.controller = PID_controller(pos_PIDs, theta_PIDs, self.motion_sleep_time)


        self.declare_parameter('xlims', [-np.inf, np.inf])
        self.declare_parameter('ylims', [-np.inf, np.inf])
        self.declare_parameter('cell_size', 0.2)

        self.xlims = self.get_parameter('xlims').get_parameter_value().double_array_value
        self.ylims = self.get_parameter('ylims').get_parameter_value().double_array_value
        self.csize = self.get_parameter('cell_size').get_parameter_value().double_value        

        """
        Motion control initializations
        """
        self.ENABLED = False
    
        self.vel_pub = self.create_publisher(Twist, "/{}/cmd_vel".format(robot_namespace), qos)


        # Obstacles are expected to be circular ones, parametrized by (loc,radius)
        self.obstacle_detector = obstacle_detector(self)

        # current control actions
        self.v = 0.0
        self.omega = 0.0

    def motion_reset(self):
        self.waypoints = np.array([])

    def ucb_callback(self, trajectory):
        self.waypoints = ros_poses2numpy(trajectory)
        self.get_logger().info(np.array2string(self.waypoints))

        self.controller.generate_trajectory(self.waypoints)

        episode_duration = trajectory.episode_duration
        self.ENABLED = True
        self.get_logger().info('Episode Starting')
        self.init_episode()

        self.episode_timer = self.create_timer(episode_duration, self.episode_end_callback)

    def episode_end_callback(self):
        self.ENABLED = False
        result = self.create_UCB_package()
        self._ucb_publisher.publish(result)
        self.get_logger().info('Episode Stopping')
        self.episode_timer.destroy()


    def init_episode(self):
        self.node_list = []
        self.reward_list = []

    def list_coefs(self,coef_dicts):
        if len(coef_dicts)==0:
            # Hard-coded values used in development.	
            C1=-0.3
            C0=0
            b=-2
            k=1
        else:
            C1=np.array([v['C1'] for v in coef_dicts])
            C0=np.array([v['C0'] for v in coef_dicts])
            b=np.array([v['b'] for v in coef_dicts])
            k=np.array([v['k'] for v in coef_dicts])
        return C1,C0,b,k


    def get_my_readings(self):
        readings = self.robot_listeners[self.robot_namespace].get_latest_readings()
        if readings is None:
            return None
        return top_n_mean(np.array(readings),4)

    def get_my_cell(self):
        loc = self.get_my_loc()
        loc[0] = max(min(self.xlims[1],loc[0]), self.xlims[0])
        loc[1] = max(min(self.ylims[1],loc[1]), self.ylims[0])
        col_cell = -round(loc[0]/self.csize) 
        row_cell = -round(loc[1]/self.csize)
        return (row_cell, col_cell)
    

    def get_my_loc(self):

        return self.robot_listeners[self.robot_namespace].get_latest_loc()
    
    def get_my_yaw(self):
        return self.robot_listeners[self.robot_namespace].get_latest_yaw()
    
    def get_my_coefs(self):
    
        return self.robot_listeners[self.robot_namespace].get_coefs()




    def create_UCB_package(self):
        node_intervals = {}
        node_means = {}

        for i, node in enumerate(self.node_list):
            if i == 0 or node != self.node_list[i-1]:
                # This is a new node. Should not have already been visited
                try:
                    assert(node not in node_intervals)
                except:
                    self.get_logger().info(f"Node {node} already visited")

                node_intervals[node] = [i, -1]

                if i > 0:
                    assert(node != self.node_list[i-1])
                    assert(node_intervals[self.node_list[i-1]][1] == -1)
                    node_intervals[self.node_list[i-1]][1] = i

        node_intervals[self.node_list[-1]][1] = len(self.node_list)
        self.reward_list = np.array(self.reward_list, dtype = float)

        for node in node_intervals:
            node_means[node] = np.nanmean(self.reward_list[node_intervals[node][0]:node_intervals[node][1]])

        return dict2UCBPackage(node_intervals, node_means)


    def reward_callback(self):
        """ 
                Collect rewards 
        """
        if self.ENABLED and self.get_my_loc() is not None:
            self.node_list.append(self.get_my_cell())
            self.reward_list.append(self.get_my_readings())			            
            # self.get_logger().info(f"Current Cell --> {self.get_my_cell()} && Current Readings --> {self.reward_list[-1]}")
        else:
            pass
    
    def motion_callback(self):
        """
            Motion Control
        """
        if self.ENABLED:
            loc = self.get_my_loc()
            yaw = self.get_my_yaw()
            self.get_logger().info(f"Current Location --> {loc}")
            self.get_logger().info(f"Current Yaw --> {yaw}")

            if (not loc is None) and (not yaw is None):
                [v, omega] = self.controller.update(loc, yaw)                    
                [v,omega] = bounded_change_update(v,omega,self.v,self.omega) 
                
                vel_msg = turtlebot_twist(v,omega)

                # Update current v and omega
                self.v = v
                self.omega = omega
                self.vel_pub.publish(vel_msg)
            else:
                self.vel_pub.publish(stop_twist())

                # Update current v and omega
                self.v = 0.0
                self.omega = 0.0

        else:
            loc = self.get_my_loc()
            yaw = self.get_my_yaw()
            self.get_logger().info(f"Current Location --> {loc}")
            self.get_logger().info(f"Current Yaw --> {yaw}")
            self.vel_pub.publish(stop_twist())

            self.motion_reset()
        
        
    



def main(args=sys.argv):
    rclpy.init(args=args)
    args_without_ros = rclpy.utilities.remove_ros_args(args)

    arguments = len(args_without_ros) - 1
    position = 1


    # Get the robot name passed in by the user
    robot_namespace=''
    if arguments >= position:
        robot_namespace = args_without_ros[position]
        
        
    an = agent_node(robot_namespace)
    
    an.get_logger().info(str(args_without_ros))
    try:
        print('Agent Node Up')
        rclpy.spin(an)
    except KeyboardInterrupt:
        print("Keyboard Interrupt. Shutting Down...")
        for _ in range(30):# Publish consecutive stop twist for 3 seconds to ensure the robot steps.
            an.vel_pub.publish(stop_twist())
            time.sleep(0.1)
    finally:
        an.destroy_node()
        print('Agent Node Down')
        rclpy.shutdown()




if __name__ == '__main__':
    main()