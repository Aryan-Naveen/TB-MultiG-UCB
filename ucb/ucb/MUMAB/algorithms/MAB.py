import argparse
import os
from abc import ABC, abstractmethod

import seaborn as sns
import numpy as np
import random
import networkx as nx
from networkx import bipartite
import sys
import os
import json
import time
import gurobipy as gp
from tqdm import tqdm

from .utils import optimal_distribution

from ast import literal_eval

class MAB:
    def __init__(self, G, T, K, M, cell_size, xlims, ylims, agents):
        self.G = G
        self.T = T
        self.K = K
        self.M = M
        self.csize = cell_size
        self.xlims = xlims
        self.ylims = ylims
        self.agents = agents


    def compute_start_loc(self, pos):
        start = np.array([0, 0])

        dist_x = round((self.xlims[1] - self.xlims[0]), 1)
        ncells_x = (int(dist_x/self.csize) + 1)/2
        dist_y = round((self.ylims[1] - self.ylims[0]), 1)
        ncells_y = (int(dist_y/self.csize) + 1)/2
        
        for i in range(int(self.M/2)):
            for j in range(int(self.M/2)):
                start_ = np.array([-i*ncells_y*self.csize, -j*ncells_x*self.csize])
                if np.linalg.norm(start_ - pos) < np.linalg.norm(start - pos):
                    start = start_
        return start

    def _initialize_trajectory(self, loc0):
        """
        Initializes the approximations for each vertex by having agents visit all nodes. Generate trajectory for agent based on starting node
        """
        dist_x = round((self.xlims[1] - self.xlims[0]), 1)
        ncells_x = int((int(dist_x/self.csize) + 1)/2)
        dist_y = round((self.ylims[1] - self.ylims[0]), 1)
        ncells_y = int((int(dist_y/self.csize) + 1)/2)

        startrow = int(loc0[0]/self.csize)
        startcol = int(loc0[1]/self.csize)

        trajectory = []
        for row in range(startrow, startrow - ncells_y, -1):
            if (row -startrow) % 2 == 0:
                for col in range(startcol, startcol - ncells_x, -1):
                    trajectory.append([row, col])
            else:
                for col in range(startcol - ncells_x  + 1 , startcol + 1):
                    trajectory.append([row, col])

        return self.csize*np.array(trajectory)

    def process_episode_rewards(self, agent_packages, time):

        for node in self.G:
            self.G.nodes[node]['arm'].update_attributes(agent_packages, time)

        for node in self.G:
            distribution, _ = optimal_distribution([self.G.nodes[node]['arm'] for node in self.G], self.M)            

        sampled_nodes = []
        for node in self.G:
            for times in range(round(distribution[f"x_{self.G.nodes[node]['arm'].id}"])):
                sampled_nodes.append(node)

        sorted_by_pulls = sorted(sampled_nodes, key = lambda x : self.G.nodes[x]['arm'].num_pulls)
        baseline_arm = sorted_by_pulls[0]
        baseline_pulls = self.G.nodes[baseline_arm]['arm'].num_pulls

        max_ucb = max([self.G.nodes[node]['arm'].ucb for node in sampled_nodes])

        # Initialize edge weights given UCB estimate of each arm.
        # Edge weights are (max_ucb - ucb) where max_ucb is the UCB of the optimal arm
        G_directed = nx.DiGraph(self.G)
        for (u, v) in self.G.edges():
            # Floating point errors incured so flooring at 0
            G_directed.edges[u, v]["weight"] = max(max_ucb - self.G.nodes[v]['arm'].ucb, 0)
            G_directed.edges[v, u]["weight"] = max(max_ucb - self.G.nodes[u]['arm'].ucb, 0)

        # For each agent and optimal arm pair compute shortest path to create weights for bipartite graph
        # sp_dict is indexed by (agent_id, node_i) and stores a tuple (path length, actual path)
        # where path is the shortest path between the current node of the agent and the destination node
        sp_dict = {}
        for namespace, agent in self.agents.items():
            # Compute single source shortest path to all other nodes
            try:
                shortest_path        = nx.shortest_path(G_directed, source = agent.current_node(), weight = "weight") 
            except:
                for (u, v) in G_directed.edges():
                    if G_directed.edges[u, v]["weight"] < 0:
                        print(G_directed.edges[u, v]["weight"])
                assert(False)

            # Compute single source shortest path length to all other nodes
            shortest_path_length = nx.shortest_path_length(G_directed, source = agent.current_node(), weight = "weight")
            # And then add path to shortest path dictionary for all destination nodes
            for i, dest_node in enumerate(sampled_nodes):
                sp_dict[(namespace, f"{dest_node}_{i}")] = (shortest_path_length[dest_node], shortest_path[dest_node])

        # Create bipartite graph
        B = nx.Graph()
        B.add_nodes_from([('agent', namespace) for namespace, agent in self.agents.items()])
        B.add_nodes_from([(f'node_{i}', node) for i, node in enumerate(sampled_nodes)])
        for (agent_name, dest_node_str) in sp_dict:
            dest_node = literal_eval(dest_node_str.split('_')[0])

            index    = int(dest_node_str.split('_')[1])
            B.add_edge(('agent', agent_name), (f'node_{index}', dest_node), weight = sp_dict[(agent_name, dest_node_str)][0])
        assignments = bipartite.minimum_weight_full_matching(B, top_nodes = [('agent', namespace) for namespace, agent in self.agents.items()], weight = "weight")


        # Create list paths where paths[i] is the path for agent i
        paths = {}

        baseline_agent = None
        time_per_transition = 7.0

        for namespace, agent in self.agents.items():
            (node_name, dest_node) = assignments[('agent', namespace)]
            index  = int(node_name.split('_')[1])
            paths[namespace] = sp_dict[(namespace, f"{dest_node}_{index}")][1]
            agent.set_target_path(paths[namespace])
            if dest_node == baseline_arm:
                if not baseline_agent or agent.get_path_len() < baseline_agent.get_path_len():
                    baseline_agent = agent                

        episode_max_length = time_per_transition*baseline_agent.get_path_len() + 10*baseline_pulls - 1

        for namespace, agent in self.agents.items():
            agent.start_episode(episode_max_length)

if __name__ == '__main__':
    mab = MAB(None, None, 0.2, [-3.0, 0.0], [-3.0, 0.0])
    print(mab._initialize_trajectory([0.0, 0.0]))