import json, os, random
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score
from scipy.special import rel_entr
import math
import networkx as nx
from networkx.exception import NetworkXError
import community as community_louvain
from networkx.algorithms.community import girvan_newman
import itertools
from utils import *

class Reward_Manager:
    def __init__(self, graph):
        self._graph = graph
        self._org_lcc_size = 0
        self._org_edge_flows = {}
        self._prev_spearmanr = 1.0

    def compute_reward(self, G, removed_edge=None):
        """
        Compute the total reward based on three factors:
        - Connectivity (LCC size)
        - Flow retention (flow through removed edge)
        - Pagerank
        """
        cur_reward_conn = self._compute_connectivity_reward(G)
        cur_reward_flow = self._compute_flow_retention_reward(removed_edge)
        cur_reward_pagerank = self._compute_spearman_reward(G)

        return cur_reward_conn + cur_reward_flow + cur_reward_pagerank

    def setup(self):
        self._setup_connectivity()
        self._setup_flow()
        self._setup_pr()

    def reset(self):
        self.setup()

    def _setup_connectivity(self):
        self.undirected_graph = self._graph.to_undirected()
        self._org_lcc_size = len(max(nx.connected_components(self.undirected_graph), key=len))

    def _setup_flow(self):
        self._org_edge_flows = {edge: self._graph.get_edge_data(*edge)['weight'] for edge in self._graph.edges}

    def _compute_connectivity_reward(self, G):
        undirected_G = G.to_undirected()
        lcc_size_original = self._org_lcc_size
        lcc_size_new = len(max(nx.connected_components(undirected_G), key=len))
        reward_conn = -((lcc_size_new - lcc_size_original) / lcc_size_original)
        return reward_conn

    def _compute_flow_retention_reward(self, removed_edge):
        if removed_edge is None:
            return 0
        flow_removed_edge = self._org_edge_flows.get(removed_edge, 0)
        reward_flow = -flow_removed_edge
        return reward_flow

    def _setup_pr(self):
        self._org_pr = list(nx.pagerank(self._graph, tol=1e-4).values())
        self._prev_spearmanr = 1.0

    def compute_sparmanr(self, G):
        cur_spearmanr = pr_corr(G, self._graph)
        return cur_spearmanr

    def _compute_spearman_reward(self, G):
        cur_spearmanr = self.compute_sparmanr(G)
        reward = cur_spearmanr - self._prev_spearmanr
        self._prev_spearmanr = cur_spearmanr
        return reward

    