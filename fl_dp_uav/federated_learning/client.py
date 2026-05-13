"""
FL client with DP-SGD local training.

DP-SGD (Abadi et al., 2016, "Deep Learning with Differential Privacy"):
  - Per-sample gradient clipping bounds sensitivity.
  - Gaussian noise injection achieves (ε, δ)-DP per round.

Per-sample gradients are computed via a microbatch loop (batch_size=1 inner loop).
For production throughput, replace with torch.vmap or the Opacus library.
"""

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


class FLClient:
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        dataset: Subset,
        lr: float,
        local_epochs: int,
        batch_size: int,
        dp_enabled: bool = True,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.1,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.dataset = dataset
        self.lr = lr
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.dp_enabled = dp_enabled
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.device = device

    @property
    def num_samples(self) -> int:
        return len(self.dataset)

    @property
    def sample_rate(self) -> float:
        return self.batch_size / max(self.num_samples, 1)

    def set_weights(self, global_weights: Dict):
        self.model.load_state_dict(copy.deepcopy(global_weights))

    def train(self) -> Tuple[Dict, float]:
        """
        Run local training and return (updated_state_dict, avg_train_loss).
        Uses DP-SGD when dp_enabled=True.
        """
        self.model.train()
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()

        total_loss, total_steps = 0.0, 0

        for _ in range(self.local_epochs):
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                if self.dp_enabled:
                    loss = self._dp_step(x_batch, y_batch, optimizer, criterion)
                else:
                    optimizer.zero_grad()
                    loss_t = criterion(self.model(x_batch), y_batch)
                    loss_t.backward()
                    optimizer.step()
                    loss = loss_t.item()

                total_loss += loss
                total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)
        return copy.deepcopy(self.model.state_dict()), avg_loss

    def _dp_step(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """
        DP-SGD batch step:
          For each sample: forward, backward, clip gradient.
          Then sum, add Gaussian noise, average, update.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        n = x_batch.size(0)
        summed_grads = [torch.zeros_like(p) for p in params]
        total_loss = 0.0

        for xi, yi in zip(x_batch, y_batch):
            optimizer.zero_grad()
            loss = criterion(self.model(xi.unsqueeze(0)), yi.unsqueeze(0))
            loss.backward()
            total_loss += loss.item()

            # Per-sample gradient clipping
            per_sample = [p.grad.detach().clone() for p in params]
            norm = torch.sqrt(sum(g.norm() ** 2 for g in per_sample))
            scale = min(1.0, self.max_grad_norm / (norm.item() + 1e-8))
            for i, g in enumerate(per_sample):
                summed_grads[i] += g * scale

        # Add calibrated Gaussian noise, then average
        sigma = self.noise_multiplier * self.max_grad_norm
        optimizer.zero_grad()
        for p, sg in zip(params, summed_grads):
            p.grad = (sg + torch.randn_like(sg) * sigma) / n

        optimizer.step()
        return total_loss / n
