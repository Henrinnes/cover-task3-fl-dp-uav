"""
FedAvg aggregation server (McMahan et al., 2017).
Aggregates client model updates as a weighted average (weight = # local samples).
"""

import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FLServer:
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.round = 0

    def get_weights(self) -> Dict:
        return copy.deepcopy(self.model.state_dict())

    def aggregate(self, updates: List[Tuple[Dict, int]]) -> Dict:
        """
        FedAvg: new_w = Σ (n_k / N) * w_k
        updates: list of (state_dict, num_samples) from each participant.
        """
        if not updates:
            return self.get_weights()

        total = sum(n for _, n in updates)
        aggregated: Dict = {}

        for key in updates[0][0]:
            aggregated[key] = sum(
                w[key].detach().cpu().float() * (n / total) for w, n in updates
            )

        self.model.load_state_dict(aggregated)
        self.round += 1
        return copy.deepcopy(aggregated)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Returns (average_loss, accuracy)."""
        self.model.eval()
        criterion = nn.CrossEntropyLoss(reduction="sum")
        total_loss, correct, n = 0.0, 0, 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            total_loss += criterion(logits, y).item()
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)

        return total_loss / max(n, 1), correct / max(n, 1)
