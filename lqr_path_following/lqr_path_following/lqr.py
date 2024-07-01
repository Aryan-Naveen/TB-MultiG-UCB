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
from rclpy.qos import QoSProfile
from rclpy.node import Node



tools_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(tools_root))

# General dependencies
from ros2_utils.robot_listener import robot_listener
from ros2_utils.pose import prompt_pose_type_string
from ros2_utils.misc import get_sensor_names

from ucb_interfaces.msg import Observation, Rewards

# Waypoint planning dependencies
from motion_control import WaypointPlanning

from motion_control.WaypointTracking import BURGER_MAX_LIN_VEL

from util_func import analytic_dLdp, joint_F_single, top_n_mean

# Motion control dependencies
from ros2_utils.pose import bounded_change_update, turtlebot_twist, stop_twist

from motion_control.WaypointTracking import LQR_for_motion_mimicry

from collision_avoidance.obstacle_detector import obstacle_detector, source_contact_detector, boundary_detector
from collision_avoidance.regions import RegionsIntersection



COEF_NAMES = ['C1','C0','k','b']

def get_control_action(waypoints,curr_x):
    if len(waypoints)==0:
        return []
    
    waypoints = waypoints[np.argmin(np.linalg.norm(waypoints-curr_x[:2],axis=1)):]
    planning_dt = 0.5

    Q = np.array([[10,0,0],[0,10,0],[0,0,1]])
    R = np.array([[10,0],[0,10]])
    uhat,_,_ = LQR_for_motion_mimicry(waypoints,planning_dt,curr_x,Q=Q,R=R)
    return uhat

class agent_node(Node):
    def __init__(self,robot_namespace,pose_type_string,neighborhood_namespaces=None,xlims=[-np.inf,np.inf],ylims = [-np.inf,np.inf]):
        super().__init__(node_name = 'agent_node', namespace = robot_namespace)
        self.pose_type_string = pose_type_string
        self.robot_namespace = robot_namespace

        assert(robot_namespace in neighborhood_namespaces)
        if neighborhood_namespaces is None:
            self.neighborhood_namespaces = get_sensor_names(self)
        else:
            self.neighborhood_namespaces = neighborhood_namespaces

        self.robot_listeners = {namespace:robot_listener(self,namespace,self.pose_type_string)\
                         for namespace in neighborhood_namespaces}



        qos = QoSProfile(depth=10)

        """
        Episode Reward Tracking Initialization
        """
        self.episode_reward = Rewards()
        self.init_reward_msg()

        """
        Timer initialization
        """

        self.reward_sleep_time = 1e-2
        
        self.reward_collection_timer = self.create_timer(self.reward_sleep_time,self.reward_callback)

        self.motion_sleep_time = 5e-1

        self.motion_timer = self.create_timer(self.motion_sleep_time,self.motion_callback)

        """ 
        Waypoint planning initializations 
        """
            
        # Temporary hard-coded waypoints used in devel.	
        self.waypoints = np.array([[-2.5, -2.0], [-2.2, -2.0], [-2.0, -2.0]])
        self.waypoint_pub = self.create_publisher(Float32MultiArray,'waypoints',qos)
    
        """
        Motion control initializations
        """
        self.ENABLED = True
        self.enable_sub = self.create_subscription(Bool,'/MISSION_CONTROL/ENABLE',self.ENABLE_CALLBACK,qos)
    
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', qos)

        self.control_actions = deque([])

        # Obstacles are expected to be circular ones, parametrized by (loc,radius)
        self.obstacle_detector = obstacle_detector(self)
        self.source_contact_detector = source_contact_detector(self)
        self.boundary_detector = boundary_detector(self,xlims,ylims)

        # current control actions
        self.v = 0.0
        self.omega = 0.0



    
    def waypoint_reset(self):
        self.waypoints = []
    
    def motion_reset(self):
        self.control_actions = deque([])
        self.v = 0.0
        self.omega = 0.0


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
    
    def get_my_loc(self):

        return self.robot_listeners[self.robot_namespace].get_latest_loc()
    
    def get_my_yaw(self):
        return self.robot_listeners[self.robot_namespace].get_latest_yaw()
    
    def get_my_coefs(self):
    
        return self.robot_listeners[self.robot_namespace].get_coefs()

    def init_reward_msg(self):
        self.episode_reward.observations = []


    def ENABLE_CALLBACK(self,data):

        if not self.ENABLED == data.data:
            if data.data:
                self.get_logger().info('Episode Starting')
                self.init_reward_msg()
            else:
                self.get_logger().info('Episode Stopping')

        self.ENABLED = data.data


    def update_reward_msg(self):
        loc = self.get_my_loc()
        reading = self.get_my_readings()

        if loc is None or reading is None:
            return

        [x, y] = self.get_my_loc()
        self.get_logger().info('Reward collected --> loc:{} yaw:{}'.format(loc,reading))


        obs = Observation()
        obs.x = x
        obs.y = y
        obs.observation = reading
        self.episode_reward.append(obs)


    def reward_callback(self):
        """ 
                Collect rewards 
        """
        if self.ENABLED:
            self.update_reward_msg()			            
        else:
            pass
    
    def reached_end_node(self, thresh=0.1):
        curr = self.get_my_loc()
        xf = self.waypoints[-1]
        return np.linalg.norm(curr - xf) < thresh

    def motion_callback(self):
        """
            Motion Control
        """
        if self.ENABLED:
            if self.source_contact_detector.contact():
                self.vel_pub.publish(stop_twist())
            else:

                # Project waypoints onto obstacle-free spaces.
                
                free_space = RegionsIntersection(self.obstacle_detector.get_free_spaces() + self.boundary_detector.get_free_spaces() )

                loc = self.get_my_loc()
                yaw = self.get_my_yaw()
                self.get_logger().info(f"Current Location --> {loc}")
                if len(self.waypoints)==0:
                    self.get_logger().info("Running out of waypoints.")

                if (not loc is None) and (not yaw is None) and len(self.waypoints)>0:
                    curr_x = np.array([loc[0],loc[1],yaw])		
                    wp_proj = free_space.project_point(self.waypoints)
                    self.control_actions = deque(get_control_action(wp_proj,curr_x))
                    waypoint_out = Float32MultiArray()
                    waypoint_out.data = list(wp_proj.flatten())
                    self.waypoint_pub.publish(waypoint_out)

                if len(self.control_actions)>0 and not self.reached_end_node():

                    # Pop and publish the left-most control action.

                    [v,omega] = self.control_actions.popleft()
                    
                    [v,omega] = bounded_change_update(v,omega,self.v,self.omega) # Get a vel_msg that satisfies max change and max value constraints.
                    
                    vel_msg = turtlebot_twist(v,omega)

                    # Update current v and omega
                    self.v = v
                    self.omega = omega
                    print(v)
                    self.vel_pub.publish(vel_msg)
                else:
                    self.vel_pub.publish(stop_twist())

                    # Update current v and omega
                    self.v = 0.0
                    self.omega = 0.0

        else:
            self.vel_pub.publish(stop_twist())

            self.motion_reset()
        
        
    



def main(args=sys.argv):
    rclpy.init(args=args)
    args_without_ros = rclpy.utilities.remove_ros_args(args)

    print(args_without_ros)
    arguments = len(args_without_ros) - 1
    position = 1


    # Get the robot name passed in by the user
    robot_namespace=''
    if arguments >= position:
        robot_namespace = args_without_ros[position]
    
    if arguments >= position+1:
        pose_type_string = args_without_ros[position+1]
    else:
        pose_type_string = prompt_pose_type_string()
    
    if arguments >= position+2:
        neighborhood = set(args_without_ros[position+2].split(','))
    else:
        neighborhood = set(['MobileSensor{}'.format(n) for n in range(5,6)])
        # neighborhood = set(['MobileSensor2'])
    
    
    x_max = 3
    x_min = -4
    y_max = 4
    y_min = -4
    
    an = agent_node(robot_namespace,pose_type_string, neighborhood_namespaces = neighborhood,
                            xlims=[x_min-1,x_max+1],
                            ylims=[y_min-1,y_max+1])
    
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