"""Q-network for DQN-based UAV participation agents."""
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Small MLP that maps UAV state → Q-values over discrete participation actions.

    State:  [battery, snr_norm, data_quality, round_progress, global_acc, prev_reward]
    Output: Q(s, a) for each action in {skip, 1ep, 3ep, 5ep}
    """

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
