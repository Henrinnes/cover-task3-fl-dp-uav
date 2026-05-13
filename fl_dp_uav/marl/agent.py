"""
DQN agent for each UAV in the MARL-FL framework.

Each UAV independently learns a participation policy:
  State : [battery, snr_norm, data_quality, round_progress, global_acc, prev_reward]
  Actions: 0=skip, 1=1 epoch, 2=3 epochs, 3=5 epochs

Uses a target network (double-DQN style) for stable Q-value bootstrapping.
Q-networks are periodically federated across agents via FedAvg in MARLFLTrainer.
"""
import copy
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .dqn import QNetwork
from .replay_buffer import ReplayBuffer

ACTION_EPOCHS: List[int] = [0, 1, 3, 5]


class DQNAgent:
    def __init__(
        self,
        agent_id: int,
        state_dim: int = 6,
        n_actions: int = 4,
        lr: float = 1e-3,
        gamma: float = 0.95,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: float = 0.99,
        buffer_capacity: int = 2000,
        batch_size: int = 32,
        target_update_interval: int = 10,
        device: str = "cpu",
    ):
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.device = device
        self._update_count = 0

        self.q_net     = QNetwork(state_dim, n_actions).to(device)
        self.target_net = copy.deepcopy(self.q_net).to(device)
        self.optimizer  = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer     = ReplayBuffer(buffer_capacity)

    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy action selection."""
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q_net(s).argmax(dim=1).item())

    def push(self, state, action: int, reward: float, next_state, done: float):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> float:
        """One Huber-loss DQN gradient step. Returns 0.0 if buffer not ready."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        q_vals  = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q  = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self._update_count += 1
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

        if self._update_count % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    # ------------------------------------------------------------------ #
    # Federated Q-network exchange
    # ------------------------------------------------------------------ #

    def get_q_weights(self) -> Dict[str, torch.Tensor]:
        return copy.deepcopy(self.q_net.state_dict())

    def set_q_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Load federated-averaged weights into both online and target networks."""
        self.q_net.load_state_dict(copy.deepcopy(state_dict))
        self.target_net.load_state_dict(copy.deepcopy(state_dict))
