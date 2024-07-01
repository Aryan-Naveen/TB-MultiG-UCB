import numpy as np
import random
import math

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
    def __init__(self, id):
        self.id             :np.array   = id
        self.num_pulls      :int   = 0                               # Number of pulls, to be used when calculating confidence radius
        self.num_samples    :int   = 0                               # Number of samples, to be used when calculating mean reward               
        self.total_reward   :int   = 0
        self.estimated_mean :int   = 0
        self.conf_radius    :int   = 0
        self.ucb            :int   = 0
    
    def update_attributes(self, agents, time):
        raise("Not implemented!!")
    
    def reset(self):
        self.num_pulls = 0
        self.total_reward = 0
        self.estimated_mean = 0
        self.conf_radius = 0
        self.ucb = 0
        self.num_samples = 0