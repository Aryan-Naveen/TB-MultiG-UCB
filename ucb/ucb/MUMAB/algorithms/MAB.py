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

class MAB:
    def __init__(self, G, T, K, M, cell_size, xlims, ylims):
        self.G = G
        self.T = T
        self.K = K
        self.M = M
        self.csize = cell_size
        self.xlims = xlims
        self.ylims = ylims

    def compute_start_loc(self, pos):
        start = np.array([0, 0])
        for i in range(int(self.M/2)):
            for j in range(int(self.M/2)):
                start_ = np.array([i*self.ylims[0]/2, j*self.xlims[0]/2])
                if np.linalg.norm(start_ - pos) < np.linalg.norm(start - pos):
                    start = start_
        return start



    def _initialize_trajectory(self, loc0):
        """
        Initializes the approximations for each vertex by having agents visit all nodes. Generate trajectory for agent based on starting node
        """
        dist_x = round((self.xlims[1] - self.xlims[0])/2)
        ncells_x = int(dist_x/self.csize)
        dist_y = round((self.ylims[1] - self.ylims[0])/2)
        ncells_y = int(dist_y/self.csize)

        startrow = int(loc0[0]/self.csize)
        startcol = int(loc0[1]/self.csize)

        trajectory = []
        for row in range(startrow, startrow - ncells_y, -1):
            if row % 2 == 0:
                for col in range(startcol, startcol - ncells_x, -1):
                    trajectory.append([row, col])
            else:
                for col in range(startcol - ncells_x + 1, startcol + 1):
                    trajectory.append([row, col])



        return self.csize*np.array(trajectory)
    
if __name__ == '__main__':
    mab = MAB(None, None, 0.2, [-3.0, 0.0], [-3.0, 0.0])
    print(mab._initialize_trajectory([0.0, 0.0]))