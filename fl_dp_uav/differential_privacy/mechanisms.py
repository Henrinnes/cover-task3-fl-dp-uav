"""
DP-SGD gradient privatization (Abadi et al., 2016):
  1. Clip each per-sample gradient to L2 norm <= max_grad_norm  (bounding sensitivity)
  2. Sum clipped gradients, add isotropic Gaussian noise N(0, (σ·C)²·I)
  3. Average over the batch
"""

import torch
from typing import List


def clip_gradients(grads: List[torch.Tensor], max_norm: float) -> List[torch.Tensor]:
    """Per-sample L2 clipping."""
    total_norm = torch.sqrt(sum(g.norm() ** 2 for g in grads))
    scale = min(1.0, max_norm / (total_norm.item() + 1e-8))
    return [g * scale for g in grads]


def add_gaussian_noise(
    grads: List[torch.Tensor], sensitivity: float, noise_multiplier: float
) -> List[torch.Tensor]:
    sigma = noise_multiplier * sensitivity
    return [g + torch.randn_like(g) * sigma for g in grads]


def privatize_gradients(
    per_sample_grads: List[List[torch.Tensor]],
    max_grad_norm: float,
    noise_multiplier: float,
) -> List[torch.Tensor]:
    """
    Full DP-SGD batch privatization.
    per_sample_grads: list over samples, each a list of param-level tensors.
    Returns averaged noisy gradients (one tensor per parameter).
    """
    if not per_sample_grads:
        raise ValueError("per_sample_grads is empty")

    n_params = len(per_sample_grads[0])
    summed = [torch.zeros_like(per_sample_grads[0][i]) for i in range(n_params)]

    for sample_grads in per_sample_grads:
        clipped = clip_gradients(sample_grads, max_grad_norm)
        for i, g in enumerate(clipped):
            summed[i] += g

    noised = add_gaussian_noise(summed, max_grad_norm, noise_multiplier)
    n = len(per_sample_grads)
    return [g / n for g in noised]
