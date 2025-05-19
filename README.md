# STGS: Spatio-temporal Graph Sparsification

This repository contains the anonymous implementation of **STGS** (Spatio-temporal Graph Sparsification), a reinforcement learning-based framework for sparsifying spatio-temporal graphs while preserving key structural and temporal properties.

## Overview

STGS is designed to:
- Reduce the size of dynamic graphs by pruning non-essential edges.
- Preserve structural integrity (e.g., community structure, PageRank).
- Maintain downstream task performance (e.g., traffic forecasting).

The method combines reinforcement learning (Dueling DQN) with a temporal graph encoder (GCN + GRU + attention).

## Structure

- `models/`: Core architecture (Q-network, encoder, etc.)
- `utils/`: Data loading, evaluation metrics, helper functions
- `scripts/`: Training and evaluation scripts
- `data/`: Sample configuration or preprocessed datasets (if included)

## Requirements

- Python 3.8+
- PyTorch >= 1.11
- NetworkX
- scikit-learn
- NumPy
- tqdm

Install dependencies using:

```bash
pip install -r requirements.txt

