
import numpy as np
import torch
import pickle
import networkx as nx
import ssl
import torch.nn as nn
from torch.optim import Adam
from community import community_louvain
from torch_geometric.data import Data
from torch_geometric_temporal.dataset import WikiMathsDatasetLoader, METRLADatasetLoader, PemsBayDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import matplotlib.pyplot as plt
import torch.nn.functional as F
import powerlaw
from config import *
import random
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from scipy.stats import entropy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_test_loader(dataset, data, ratio):
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=ratio)
    train_input = np.array(train_dataset.features) 
    train_target = np.array(train_dataset.targets) 

    if len(train_input.shape) == 3:
        train_input = np.expand_dims(train_input, axis=2)
        train_target = np.expand_dims(train_target, axis=2)
        
    if data == 'pems-bay':
       train_target = train_target.reshape(-1, train_target.shape[1], train_target.shape[2]*train_target.shape[3])

    train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor)  # (B, N, T)
    train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=config['batch_size'], shuffle=True, drop_last=True)

    test_input = np.array(test_dataset.features) 
    test_target = np.array(test_dataset.targets) 

    if len(test_input.shape) == 3:
        test_input = np.expand_dims(test_input, axis=2)
        test_target = np.expand_dims(test_target, axis=2)
    
    if data == 'pems-bay':
        test_target = test_target.reshape(-1, test_target.shape[1], test_target.shape[2]*test_target.shape[3])

    test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor)  # (B, N, T)
    test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=config['batch_size'], shuffle=True, drop_last=True)

    return train_loader, test_loader, train_x_tensor, test_x_tensor


def import_dataset(dataset_name):
    if dataset_name == 'pems-bay':
        loader = PemsBayDatasetLoader()
        dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=1)
    elif dataset_name == 'metr-la':
        loader = METRLADatasetLoader()
        dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=1)
    elif dataset_name == 'wiki':
        loader = WikiMathsDatasetLoader()
        dataset = loader.get_dataset(12)
    elif dataset_name == 'covid':
        dataset = torch.load('data/covid.pt')
    else:
        raise ValueError('Dataset not found')
    
    datasets = list(dataset)
    
    return dataset, datasets


def community_density(G):
    # Detect communities using the Louvain method
    partition = community_louvain.best_partition(G)

    # Get the number of nodes in each community
    community_nodes = {}
    for node, comm in partition.items():
        if comm not in community_nodes:
            community_nodes[comm] = []
        community_nodes[comm].append(node)

    # Calculate the number of inter-community edges
    inter_edges = 0
    for edge in G.edges():
        if partition[edge[0]] != partition[edge[1]]:
            inter_edges += 1

    # Calculate the number of possible inter-community edges
    possible_inter_edges = 0
    communities = list(community_nodes.values())
    for i in range(len(communities)):
        for j in range(i + 1, len(communities)):
            possible_inter_edges += len(communities[i]) * len(communities[j])

    # Compute the inter-community density
    inter_community_density = inter_edges / possible_inter_edges
    
    return inter_community_density

def averge_shortest_path_length(G):
    return nx.average_shortest_path_length(G)

def powerlaw_exponent(G):
    degrees = [degree for node, degree in G.degree()]
    fit = powerlaw.Fit(degrees)
    alpha = fit.power_law.alpha
    sigma = fit.power_law.sigma
    
    return alpha, sigma

def graph_stats(graph):
    print('Number of nodes: ', graph.number_of_nodes())
    print('Number of edges: ', graph.number_of_edges())
    print('Average degree: ', np.mean([d for n, d in graph.degree()]))
    print('Average clustering coefficient: ', nx.average_clustering(graph))
    print('Density: ', nx.density(graph))
    print('max degree: ', max(dict(graph.degree()).values()))
    print('assortativity: ', nx.degree_assortativity_coefficient(graph))
    print('Triangle count: ', sum(nx.triangles(nx.to_undirected(graph)).values()) / 3)
    print("Inter-community Density:", community_density(nx.to_undirected(graph)))
    print("Powerlaw exponent:", powerlaw_exponent(graph))
    print('Graph Diameter:', diameter(graph))
    
    if not nx.is_connected(nx.to_undirected(graph)):
        print('Graph is not connected')
        llc = max(nx.connected_components(nx.to_undirected(graph)), key=len)
        subgraph = graph.subgraph(llc)
        print('Average shortest path length: ', averge_shortest_path_length(nx.to_undirected(subgraph)))
    else:
        print('Average shortest path length: ', averge_shortest_path_length(nx.to_undirected(graph))) 


def edge_index_to_adj(edge_index, num_nodes):
    adj = torch.zeros((num_nodes, num_nodes), dtype=int)
    adj[edge_index[0], edge_index[1]] = 1
    return adj

def adj_to_edge_index(adj):
    # Find the indices of non-zero elements
    edge_index = adj.nonzero(as_tuple=False).t()
    return edge_index
    

def pr_corr(G1, G2):
    pagerank_G1 = nx.pagerank(G1, weight='weight')
    pagerank_G2 = nx.pagerank(G2, weight='weight')

    # Ensure both graphs have the same set of nodes
    nodes_G1 = set(G1.nodes())
    nodes_G2 = set(G2.nodes())
    common_nodes = list(nodes_G1 & nodes_G2)

    pagerank_G1_values = np.array([pagerank_G1[node] for node in common_nodes])
    pagerank_G2_values = np.array([pagerank_G2[node] for node in common_nodes])

    # Correlation Coefficient
    # correlation, _ = pearsonr(pagerank_G1_values, pagerank_G2_values)
    # print(f'Pearson correlation coefficient: {correlation}')

    spearmanr_corr, _ = spearmanr(pagerank_G1_values, pagerank_G2_values)
    
    return spearmanr_corr

def shortest_path(G):
    G = G.to_undirected()
    if not nx.is_connected(G):
        llc = max(nx.connected_components(G), key=len)
        G_subgraph = G.subgraph(llc)
        sp = nx.average_shortest_path_length(G_subgraph)
    else:
        sp = nx.average_shortest_path_length(G)
    return sp

def diameter(G):
    G = G.to_undirected()
    if not nx.is_connected(G):
        llc = max(nx.connected_components(G), key=len)
        G_subgraph = G.subgraph(llc)
        dm = nx.diameter(G_subgraph)
    else:
         dm = nx.diameter(G)
    return dm


def evaluate_model(model, static_edge_index, test_loader, device, criterion):
    model.eval()
    total_mse_loss = []
    total_rmse_loss = []
    total_mae_loss = []
    with torch.no_grad():           
        for i, (encoder_inputs, labels) in enumerate(test_loader):
            y_hat = model(encoder_inputs.to(device), static_edge_index.to(device)).to(device)
            mse = criterion(y_hat, labels.to(device))
            mae = torch.nn.L1Loss()(y_hat, labels.to(device))
            rmse = torch.sqrt(mse)
            
            total_mse_loss.append(mse.item())
            total_mae_loss.append(mae.item())
            total_rmse_loss.append(rmse.item())
            
            mae_loss = sum(total_mae_loss)/len(total_mae_loss)
            rmse_loss = sum(total_rmse_loss)/len(total_rmse_loss)
            mse_loss = sum(total_mse_loss)/len(total_mse_loss)
    
    return mae_loss, mse_loss, rmse_loss

def compute_stationary_distribution(graph):
    adjacency_matrix = nx.to_numpy_array(graph)
    degree_vector = adjacency_matrix.sum(axis=1)
    stationary_distribution = degree_vector / degree_vector.sum()
    return stationary_distribution

def compute_kl_divergence(distribution1, distribution2):
    epsilon = 1e-10
    distribution1 = np.clip(distribution1, epsilon, 1)
    distribution2 = np.clip(distribution2, epsilon, 1)
    kl_divergence = entropy(distribution1, distribution2)
    return kl_divergence

def evaluate_sparsified_graph(original_graph, sparsified_graph):
    pi_orig = compute_stationary_distribution(original_graph)
    pi_sparsified = compute_stationary_distribution(sparsified_graph)
    kl_div = compute_kl_divergence(pi_orig, pi_sparsified)
    return kl_div

def calculate_pagerank_preservation(G, G_prime):
    G_pr_d = nx.pagerank(G)
    G_prime_pr_d = nx.pagerank(G_prime)
    G_pr = list(G_pr_d.values()) 
    G_prime_pr = list(G_prime_pr_d.values())
    cur_spearmanr = spearmanr(G_pr, G_prime_pr).correlation
    return cur_spearmanr

def calculate_ari_preservation_louvain(G, G_prime):
    # convert to undirected graph
    G = G.to_undirected()
    G_prime = G_prime.to_undirected()
    
    partition_G = community_louvain.best_partition(G)
    partition_G_prime = community_louvain.best_partition(G_prime)

    labels_G = [partition_G[node] for node in G.nodes()]
    labels_G_prime = [partition_G_prime[node] for node in G_prime.nodes()]

    NMI = nmi(labels_G, labels_G_prime)
    ARI = ari(labels_G, labels_G_prime)

    return NMI, ARI