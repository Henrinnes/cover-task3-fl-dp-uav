"""
FL client with DP-SGD local training.

DP-SGD (Abadi et al., 2016, "Deep Learning with Differential Privacy"):
  - Per-sample gradient clipping bounds sensitivity.
  - Gaussian noise injection achieves (ε, δ)-DP per round.

Per-sample gradients use torch.func.vmap + grad so the full batch is processed in
a single GPU kernel (vs a Python loop over samples), giving ~30-50× speedup.
Requires GroupNorm instead of BatchNorm (no cross-sample statistics).
"""

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import grad_and_value, vmap, functional_call
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
        DP-SGD batch step using vectorised per-sample gradients (vmap + grad).

        All per-sample forward/backward passes are fused into one GPU kernel via
        torch.func.vmap, avoiding the Python-level sample loop and giving ~30-50×
        speedup over the microbatch approach.

        Steps:
          1. vmap(grad(loss)) over the batch → per_sample_grads[param][n, *shape]
          2. Compute per-sample gradient norms, derive clip coefficients.
          3. Sum clipped gradients, add Gaussian noise N(0, (σC)²I), divide by n.
          4. Write into .grad and call optimizer.step().
        """
        model = self.model
        named_params = {k: v for k, v in model.named_parameters() if v.requires_grad}
        named_buffers = dict(model.named_buffers())
        n = x_batch.size(0)

        # Model must be in eval to keep GroupNorm deterministic across vmap lanes;
        # DP noise replaces the stochasticity that Dropout would add.
        was_training = model.training
        model.eval()

        def loss_for_one(params, buffers, x, y):
            out = functional_call(model, (params, buffers), x.unsqueeze(0))
            return F.cross_entropy(out, y.unsqueeze(0))

        # grad_and_value returns (grad_dict, scalar_loss) per sample — one GPU kernel,
        # no extra forward pass after optimizer.step() (avoids CUDA misaligned-address).
        per_sample_grads, per_sample_losses = vmap(
            grad_and_value(loss_for_one), in_dims=(None, None, 0, 0)
        )(named_params, named_buffers, x_batch, y_batch)

        if was_training:
            model.train()

        # Per-sample L2 norms across all parameters
        norms_sq = sum(
            g.reshape(n, -1).norm(dim=1) ** 2 for g in per_sample_grads.values()
        )
        norms = norms_sq.sqrt()                                        # [n]
        clip_coefs = torch.clamp(self.max_grad_norm / (norms + 1e-8), max=1.0)

        # Sum clipped gradients + Gaussian noise, then average over batch
        sigma = self.noise_multiplier * self.max_grad_norm
        optimizer.zero_grad()
        for name, p in model.named_parameters():
            if p.requires_grad:
                g = per_sample_grads[name]                              # [n, *shape]
                coef = clip_coefs.reshape(n, *([1] * (g.dim() - 1)))
                clipped_sum = (g * coef).sum(dim=0)
                p.grad = (clipped_sum + torch.randn_like(clipped_sum) * sigma) / n

        optimizer.step()
        return float(per_sample_losses.mean().item())
