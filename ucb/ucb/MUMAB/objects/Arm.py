import numpy as np
import random
import math

from ucb_interfaces.msg import UCBAgentPackage

from .MultiAgentInteraction import MultiAgentInteractionInterface, ConstantMultiAgentInteraction

class Arm:
    """
    Arm class for multi-armed bandit problem

    Attributes:
        id:             np.array, unique identifier for arm (location in 2d grid)
        num_pulls:      int, number of times the arm has been pulled
        total_reward:   int, total reward accumulated from pulling arm. Only notes reward from single pull of arm
        estimated_mean: int, estimated mean of the arm
        conf_radius:    int, confidence radius of the arm
        ucb:            int, upper confidence bound of the arm
        function:       function, function used to compute multiplicative benefit of multiple agents sampling the same arm

    Methods:
        update_attributes:  Updates the attributes of the arm
        reset:              Resets the arm attributes to initial values
    """
    def __init__(self, row, col):
        self.id             :int   = self._encode_rowcol2id(row, col)
        self.num_pulls      :int   = 0                               # Number of pulls, to be used when calculating confidence radius
        self.num_samples    :int   = 0                               # Number of samples, to be used when calculating mean reward               
        self.total_reward   :int   = 0
        self.estimated_mean :int   = 0
        self.conf_radius    :int   = 0
        self.ucb            :int   = 0
        self.interaction    :MultiAgentInteractionInterface   = ConstantMultiAgentInteraction(self.id, 4)
    

    def _encode_rowcol2id(self, row, col):
        return int(row * 100 + col)
    
    def update_attributes(self, agent_packages, time):
        total_episode_reward = 0     # Total episode reward
        total_episode_counts = 0     # Total episode sampling counts

        sampling_intervals   = []    # Intervals of when agents were sampling the arm
        earliest_obs         = time  # Earliest time when an agent sampled
        latest_obs           = 0     # Latest time when an agent sampled
        total_episode_pulls  = 0     # Total number of pulls in the episode


        for namespace, pkg in agent_packages.items():
            agent_arm_intervals, agent_arm_means = self.convert_pkg2lists(pkg)
            if self.id in agent_arm_intervals and not math.isnan(agent_arm_means[self.id]):
                length = agent_arm_intervals[self.id][1] - agent_arm_intervals[self.id][0]
                mean   = agent_arm_means[self.id]

                total_episode_reward += mean * length
                total_episode_counts += length


                sampling_intervals.append(agent_arm_intervals[self.id])
                earliest_obs = min(earliest_obs, agent_arm_intervals[self.id][0])
                latest_obs   = max(latest_obs, agent_arm_intervals[self.id][1])

        for i in range(earliest_obs, latest_obs):
            for interval in sampling_intervals:
                if i >= interval[0] and i < interval[1]:
                    total_episode_pulls += 1
                    break 

        self.num_pulls     += total_episode_pulls
        self.num_samples   += total_episode_counts
        self.total_reward  += total_episode_reward
        self.estimated_mean = self.total_reward / self.num_samples
        self.conf_radius    = np.sqrt(2 * np.log(time) / self.num_pulls)
        self.ucb            = self.estimated_mean + self.conf_radius


    def convert_pkg2lists(self, pkg):
        arm_intervals = {}
        arm_means = {}
        for node in pkg:
            arm_intervals[node.id] = [node.interval[0], node.interval[1]]
            arm_means[node.id] = node.mean
        return arm_intervals, arm_means

    def reset(self):
        self.num_pulls = 0
        self.total_reward = 0
        self.estimated_mean = 0
        self.conf_radius = 0
        self.ucb = 0
        self.num_samples = 0