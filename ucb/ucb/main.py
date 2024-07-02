from .MUMAB.algorithms.MAB import MAB
from .MUMAB import objects as mobj

import numpy as np
import networkx as nx
import decimal

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.qos import QoSProfile

from .ros2_utils.robot_listener import robot_listener
from .ros2_utils.robot_publisher import robot_publisher
import time



class MAB_node(Node):
    def __init__(self):
        super().__init__(node_name = 'MAB_node')
        self.declare_parameters(
            namespace='UCB_params',
            parameters=[
                ('neighborhood_namespaces', rclpy.Parameter.Type.STRING_ARRAY),
                ('pose_type_string', rclpy.Parameter.Type.STRING),
                ('xlims', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('ylims', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('cell_size', rclpy.Parameter.Type.DOUBLE),
                ('T', rclpy.Parameter.Type.INTEGER),
                ('M', rclpy.Parameter.Type.INTEGER),
            ]
        )

        '''
        Get Grid Graph/MAB information
        '''
        self.declare_parameter('xlims', [-3.0, 0.0])
        self.declare_parameter('ylims', [-3.0, 0.0])
        self.declare_parameter('cell_size', 0.2)
        self.declare_parameter('T', 10000)
        self.declare_parameter('M', 4)

        self.xlims = self.get_parameter('xlims').get_parameter_value().double_array_value
        self.ylims = self.get_parameter('ylims').get_parameter_value().double_array_value
        self.csize = self.get_parameter('cell_size').get_parameter_value().double_value        

        self.T = self.get_parameter('T').get_parameter_value().integer_value
        self.M = self.get_parameter('M').get_parameter_value().integer_value

        num_rows = 1 + (self.ylims[1] - self.ylims[0])/self.csize
        num_cols = 1 + (self.xlims[1] - self.xlims[0])/self.csize
        self.K = int(num_cols*num_rows)   

        try:
            assert(decimal.Decimal(str(self.ylims[1] - self.ylims[0])) % decimal.Decimal(str(self.csize)) == 0.0)
            assert(decimal.Decimal(str(self.xlims[1] - self.xlims[0])) % decimal.Decimal(str(self.csize)) == 0.0)
        except:
            raise AssertionError(f"Grid Graph is not divisible by cell size")

        '''
        Initialize the grid graph
        Create empty "Arm" objects for MAB algorithm to update
        '''
        self.G = nx.grid_2d_graph(int(num_rows), int(num_cols))
        for i in self.G:
            loc = np.asarray(i)
            self.G.nodes[i]['arm'] = mobj.Arm(loc[0], loc[1])
            self.G.nodes[i]['id']  = i
            self.G.nodes[i]['prev_node'] = self.G.nodes[i]


        '''
        Get Agent controlling information
        '''
        self.declare_parameter('neighborhood_namespaces', ['MobileSensor{}'.format(n) for n in range(1,6)])
        self.neighborhood_namespaces = self.get_parameter('neighborhood_namespaces').get_parameter_value().string_array_value     

        self.declare_parameter('pose_type_string', 'optitrack')
        self.pose_type_string = self.get_parameter('pose_type_string').get_parameter_value().string_value



        self.get_logger().info(f"Initializing MAB Node with {self.T} time steps, {self.K} arms, and {self.M} agents")

        self.agents = {}

        for namespace in self.neighborhood_namespaces:
            self.agents[namespace] = mobj.Agent(namespace, robot_listener(self,namespace,self.pose_type_string), robot_publisher(self,namespace), self.csize)


        self.initialization_timer_check = 1e-1
        
        self.initialization_timer = self.create_timer(self.initialization_timer_check, self.initialization_callback)

        self.process_episode_timer_check = 1e-1

        self.process_episode_timer = self.create_timer(self.process_episode_timer_check, self.episode_ucb_callback)


        self.discrete_time = 0
        self.started = False

        self.MAB = MAB(self.G, self.T, self.K, self.M, self.csize, self.xlims, self.ylims, self.agents)



    def episode_ucb_callback(self):
        if not self.started:
            return

        self.discrete_time += 1
        for namespace, agent in self.agents.items():
            if not agent.listener.latest_episode_concluded():
                return
        
        reward_packages = {}
        for namespace, agent in self.agents.items():
            reward_packages[namespace] = agent.listener.get_latest_episode_rewards()

        self.MAB.process_episode_rewards(reward_packages, self.discrete_time)

        

    def initialization_callback(self):
        if not self._ready():
            return

        self.discrete_time = 0
        self.started = True

        locs =  self._get_robot_locs()
        for robot, pos in locs.items():
            pos0 = self.MAB.compute_start_loc(pos)
            trajectory = self.MAB._initialize_trajectory(pos0)
            self.get_logger().info(f"Publishing trajectory for {robot} that starts at {pos0} but is currently at {pos}")
            self.agents[robot].publisher.publish_trajectory(trajectory, 50.0)
        self.initialization_timer.destroy()

    def _ready(self):
        locs =  self._get_robot_locs()
        for robot, pos in locs.items():
            if pos is None:
                return False
        return True


    def _get_robot_locs(self):
        return {namespace:agent.listener.get_latest_loc() for namespace, agent in self.agents.items()}

def main(args=None):
    rclpy.init(args=args)
    mab_node = MAB_node()
    try:
        print('MAB Centralizer Node Up')
        rclpy.spin(mab_node)
    except KeyboardInterrupt:
        print("Keyboard Interrupt. Shutting Down...")
    finally:
        mab_node.destroy_node()
        print('Agent Node Down')
        rclpy.shutdown()


if __name__ == '__main__':
    main()