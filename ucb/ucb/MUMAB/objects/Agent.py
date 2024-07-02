import numpy as np

class Agent:
    def __init__(self, namespace, listener, publisher, csize):
        self.namespace = namespace
        self.listener = listener
        self.publisher = publisher
        self.csize = csize
        self.target_path = None
    
    def current_node_id(self):
        loc = self.listener.get_latest_loc()
        return self.loc2nodeid(loc)

    def current_node(self):
        loc = self.listener.get_latest_loc()
        col = -round(loc[0]/self.csize)
        row = -round(loc[1]/self.csize)
        return (row, col)

    def loc2nodeid(self, loc):
        col = -round(loc[0]/self.csize)
        row = -round(loc[1]/self.csize)
        return int(100 * row + col)

    def node2loc(self, node):
        return [-self.csize*node[1], -self.csize*node[0]]

    def set_target_path(self, path):
        self.target_path = np.array([self.node2loc(node) for node in path])

    def start_episode(self, episode_duration):
        assert(self.target_path is not None)
        self.publisher.publish_trajectory(self.target_path, episode_duration)
        self.target_path = None
    
    def get_path_len(self):
        return self.target_path.shape[0]