import torch
import argparse, os
import ssl
import logging

ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
'dataset': 'pems-bay', #'metr-la', 'pems-bay', 'wiki', 'covid'
'batch_size' : 64,
'num_epochs' : 200,
'periods': 12,
'lr' : 0.001,
'num_features' : 2,
'hidden_size' : 32,
'hidden_size_DQN' : 10,  
'reduction_rate': 0.2,
'max_epsilon': 1.0,
'min_epsilon': 0.01,
'decay_rate': 0.1, 
'gamma': 0.99,
'buffer_size': 10000,
'train_ratio': 0.8,
'alpha' : 0.6,  # Prioritization exponent
'beta' : 0.4,  # Importance sampling exponent
'encoder_path': './data/metrla_model.pth', 
'save_path': './out/qnet_model.pth',
'target_update_frequency': 10,
'evaluation_frequency': 200,
'save_frequency': 50,
}
