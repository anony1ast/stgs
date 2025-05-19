import torch
import numpy as np
import random
import pickle
import logging
from torch.optim import Adam
import torch.nn.functional as F

from config import config, device
from models import Dueling_QNet
from environment import TempoGraphEnv
from utils import import_dataset, train_test_loader, graph_stats
from replay_buffer import PrioritizedReplayBuffer

def add_experience(replay_buffer, experience, error):
    replay_buffer.add(experience, error)

def free_memory(*args):
    for arg in args:
        del arg
    torch.cuda.empty_cache()

def train():
    dataset, datasets = import_dataset(config['dataset'])
    train_loader, test_loader, x_train, x_test = train_test_loader(dataset, config['dataset'], config['train_ratio'])

    qnet = Dueling_QNet(
        config['hidden_size'], config['hidden_size_DQN'], config['num_features'], config['periods'],
        config['batch_size'], config['encoder_path'], device=device).to(device)

    target_qnet = Dueling_QNet(
        config['hidden_size'], config['hidden_size_DQN'], config['num_features'], config['periods'],
        config['batch_size'], config['encoder_path'], device=device).to(device)
    target_qnet.load_state_dict(qnet.state_dict())

    optimizer = Adam(qnet.parameters(), lr=config['lr'])
    replay_buffer = PrioritizedReplayBuffer(config['buffer_size'], alpha=config['alpha'])

    env = TempoGraphEnv(dataset, config['reduction_rate'])
    original_graph = env.graph
    graph_stats(original_graph)

    epsilon = config['max_epsilon']
    Loss = []
    # x = x_train[-config['batch_size']:].to(device)
    x = x_train[-1].unsqueeze(0).to(device)  # [1, 207, 2, 12]

    for episode in range(1, config['num_epochs'] + 1):
        logging.info(f"--- Episode {episode} started ---")
        state, _ = env.reset()
        state = torch.tensor(state).to(device)
        state = state.nonzero(as_tuple=False).t()
        done = False
        reward_total = 0

        while not done:
            edge_list = list(env.current_graph.edges())

            if np.random.rand() < epsilon:
                action = random.randint(0, len(edge_list) - 1)
            else:
                with torch.no_grad():
                    q_values = qnet(x, state, edge_list).to(device)
                    action = q_values.argmax().item()

            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.tensor(next_state).to(device)
            next_state = next_state.nonzero(as_tuple=False).t()

            with torch.no_grad():
                q_val = qnet(x, state, edge_list).to(device)[action]
                next_edge_list = list(env.current_graph.edges())
                next_q_val = target_qnet(x, next_state, next_edge_list).to(device).max()
                td_target = reward + (1 - int(done)) * config['gamma'] * next_q_val
                td_error = float(torch.clamp(td_target - q_val, -1e5, 1e5).item())

            add_experience(replay_buffer, (state, action, reward, next_state, done), error=td_error)
            state = next_state
            reward_total += reward

        if len(replay_buffer) >= config['batch_size']:
            qnet.train()
            optimizer.zero_grad()

            batch, indices, weights = replay_buffer.sample(config['batch_size'], beta=config['beta'])
            states, actions, rewards, next_states, dones = zip(*batch)

            rewards_eval = torch.tensor(rewards).to(device).float()
            dones_eval = torch.tensor(dones).to(device).float()
            weights_tensor = torch.tensor(weights).to(device).float()

            q_values_list = []
            next_q_values_list = []
            for state, next_state in zip(states, next_states):
                edge_list = list(env.current_graph.edges())
                q_values = qnet(x, state, edge_list).to(device)
                with torch.no_grad():
                    next_edge_list = list(env.current_graph.edges())
                    next_q_values = target_qnet(x, next_state, next_edge_list).to(device)

                q_values_list.append(q_values.max())
                next_q_values_list.append(next_q_values.max())

            q_values_tensor = torch.stack(q_values_list).to(device).float()
            next_q_values_tensor = torch.stack(next_q_values_list).to(device).float()

            target_q_values = rewards_eval + (1 - dones_eval) * config['gamma'] * next_q_values_tensor
            loss_unreduced = F.mse_loss(q_values_tensor, target_q_values.detach(), reduction='none')
            loss = (loss_unreduced * weights_tensor).mean()
            Loss.append(loss.item())

            td_errors = (q_values_tensor - target_q_values).detach().cpu().numpy()
            replay_buffer.update_priorities(indices, td_errors)

            loss.backward()
            optimizer.step()

            epsilon = config['min_epsilon'] + (config['max_epsilon'] - config['min_epsilon']) * np.exp(
                -config['decay_rate'] * episode)

            loss_mean = torch.mean(torch.tensor(Loss)).item()

            logging.info(
                f'Episode: {episode}, Loss: {loss_mean:.4f}, Reward: {reward_total:.4f}, Next Epsilon: {epsilon:.6f}')

            if episode % config['target_update_frequency'] == 0:
                target_qnet.load_state_dict(qnet.state_dict())

            if episode % config['save_frequency'] == 0:
                logging.info('Saving model ...')
                torch.save(qnet.state_dict(), config['save_path'])

    graph_stats(env.current_graph)
    with open('outputs/sgraph.pkl', 'wb') as f:
        pickle.dump(env.current_graph, f)

if __name__ == '__main__':
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(filename='logs.log', level=logging.INFO, format='%(message)s')

    train()

