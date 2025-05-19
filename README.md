# STGS: Spatio-temporal Graph Sparsification

This repository contains the anonymous implementation of **STGS** (Spatio-temporal Graph Sparsification), a reinforcement learning-based framework for sparsifying spatio-temporal graphs while preserving key structural and temporal properties.

## Overview

STGS is designed to:
- Reduce the size of dynamic graphs by pruning non-essential edges.
- Preserve structural integrity (e.g., community structure, PageRank).
- Maintain downstream task performance (e.g., traffic forecasting).

The method combines reinforcement learning (Dueling DQN) with a temporal graph encoder (GCN + GRU + attention).

## Structure
- `main.py` – Entry point for training and evaluation.
- `config.py` – Configuration for datasets, model parameters, and runtime settings.
- `environment.py` – MDP environment for graph sparsification.
- `models_DuelingDQN.py` – Q-network and encoder architecture.
- `replay_buffer.py` – Experience replay buffer used by the RL agent.
- `reward_manager.py` – Computes sparsification rewards (structure-based).
- `reward_manager_pred.py` – Computes prediction-oriented rewards.
- `utils.py` – Helper functions for logging, metrics, and data handling.

## Requirements
- python 3.8+
- pytorch 
- torch-geometric
- torch-geometric-temporal
- gymnasium
- networkX
- scikit-learn
- python-louvain
- powerlaw
- numpy
- panads


