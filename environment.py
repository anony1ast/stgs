import gymnasium as gym
from gymnasium import spaces
import random
import networkx as nx
import numpy as np
import community as community_louvain
from sklearn.metrics import normalized_mutual_info_score
from reward_manager import Reward_Manager


class TempoGraphEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, dataset, target_edge_rate=0.2):
        super(TempoGraphEnv, self).__init__()
        
        self.dataset = dataset  
        self.datasets = list(self.dataset)
        self.initial_edge_index = self.dataset.edge_index
        self.target_edge_rate = target_edge_rate
        self.graph = self.generate_graph()
        self.num_nodes = self.graph.number_of_nodes()
        self.initial_num_edges = self.graph.number_of_edges()
        self.target_edge_reduction = int(self.initial_num_edges * (1-self.target_edge_rate))
        print("Target Edge Reduction: ", self.target_edge_reduction)

        self.reward_man = Reward_Manager(self.graph)
    
        self.action_space = spaces.Discrete(self.initial_num_edges)
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.graph.nodes), len(self.graph.nodes)), dtype=np.float64)
    
    
    def generate_graph(self):
        # self.initial_edge_index = self.dataset.edge_index
        edges = self.initial_edge_index.T.tolist()
        G = nx.DiGraph()
        G.add_edges_from(edges)   
        print("Initial Graph: ", G) 
        self.num_nodes = G.number_of_nodes()
        
        # Add features to the nodes
        # features = np.array(self.dataset.features)
        # for i in range(self.num_nodes):
        #     G.nodes[i]['features'] = features[i]
            
        # # Add target values to the nodes
        # labels = [self.datasets[i].y for i in range(len(self.datasets))]
        # for i in range(self.num_nodes):
        #     G.nodes[i]['labels'] = labels[i]
        
        # Add edge weights
        edge_attr = self.datasets[0].edge_attr
        for i, (u, v) in enumerate(G.edges()):
            G[u][v]['weight'] = edge_attr[i].item()
        return G   
    
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.current_graph = self.graph.copy()
        self.obs = self.convert_2_adj(self.initial_edge_index)
        self.reward_man.reset()
        return self.obs, self.current_graph
    
    def convert_2_adj(self, edge_index):
        # convert edge_index to adjacency matrix (numpy)
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype = int)
        adj[edge_index[0], edge_index[1]] = 1
        return adj
    
    def step(self, action):
        reward = 0
        done = False
        
        G = nx.from_numpy_array(self.obs, create_using=nx.DiGraph)
        
        self.edges = list(G.edges(data=True))
                
        if action < len(self.edges):
            # remove edges
            edge_to_remove = self.edges[action]
            # print(edge_to_remove)
            G.remove_edge(edge_to_remove[0],edge_to_remove[1])
        
            reward = self.reward_man.compute_reward(G) 
            
            # check if the target edge reduction is reached
            if G.number_of_edges() <= self.target_edge_reduction:
                done = True 
            else:
                done = False
        else:
            # expected action is out of bounds
            print("Expected action is out of bounds")
            reward = -1
            done = False
            
        self.current_graph = G
        self.obs = nx.to_numpy_array(self.current_graph)
        
        # print("Current Graph: ", self.current_graph)
        
        return self.obs, reward, done, False, {}

    
    def render(self, mode='human', close=False):
        pass
    
    # def close(self):
    #     pass