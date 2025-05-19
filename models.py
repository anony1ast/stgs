import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Module
from torch_geometric_temporal.nn.recurrent import A3TGCN2


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class TemporalGNN(nn.Module):
    def __init__(self, node_features, periods, batch_size):
        super(TemporalGNN, self).__init__()
        self.tgnn = A3TGCN2(
            in_channels=node_features,
            out_channels=32,
            periods=periods,
            batch_size=batch_size
        )
        self.linear = nn.Linear(32, periods)
        self.apply(initialize_weights)

    def forward(self, x, edge_index):
        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h


class Dueling_QNet(Module):
    def __init__(self, hidden_size, hidden_size_DQN, node_features, periods, batch_size, encoder_path, device):
        super().__init__()
        self.device = device
        self.encoder_path = encoder_path
        self.node_encoder = TemporalGNN(node_features, periods, batch_size).to(self.device)

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * periods, hidden_size_DQN),
            nn.ReLU(),
            nn.Linear(hidden_size_DQN, hidden_size_DQN),
            nn.ReLU()
        )

        self.value_stream = Linear(hidden_size_DQN, 1)
        self.advantage_stream = Linear(hidden_size_DQN, 1)

        self.apply(initialize_weights)

    def load_encoder(self):
        try:
            self.node_encoder.load_state_dict(torch.load(self.encoder_path, map_location=self.device))
            self.node_encoder.eval()
            for param in self.node_encoder.parameters():
                param.requires_grad = False
        except RuntimeError as e:
            print(f"Failed to load encoder weights: {e}")
            raise

    def pre_train(self, x, edge_index):
        self.load_encoder()
        with torch.no_grad():
            node_embs = self.node_encoder(x, edge_index)
        return node_embs.squeeze(0)

    def forward(self, x, edge_index, edge_list):
        node_embs = self.pre_train(x, edge_index)
        # print(node_embs.shape)

        if isinstance(edge_list, list):
            edge_embeddings = [
                torch.cat([node_embs[i], node_embs[j]], dim=0) for i, j in edge_list
            ]
            edge_embeddings = torch.stack(edge_embeddings, dim=0)
        else:
            src, dst = edge_list
            edge_embeddings = torch.cat([node_embs[src], node_embs[dst]], dim=1)

        edge_features = self.edge_mlp(edge_embeddings)
        value = self.value_stream(edge_features)
        advantage = self.advantage_stream(edge_features)
        q_vals = value + (advantage - advantage.mean(dim=0, keepdim=True))
        return q_vals.squeeze(-1)
    