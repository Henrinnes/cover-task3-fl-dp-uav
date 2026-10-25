"""
Plots for MARL + FL results (python main.py --marl).

Generates a 2×2 figure:
  Top-left  : Test accuracy vs rounds (MARL-FL vs standard FL baseline)
  Top-right : Mean agent reward vs rounds
  Bottom-left: Adaptive participation rate vs rounds (vs static 60% baseline)
  Bottom-right: Privacy budget (DP-ε) consumption vs rounds

Output: results/marl/marl_results.pdf
"""
import json
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _smooth(vals, window: int = 5):
    if len(vals) <= window:
        return vals
    return np.convolve(vals, np.ones(window) / window, mode="valid").tolist()


def plot_marl_results(
    marl_path: str,
    fl_path: Optional[str] = None,
    output_dir: str = "./results/marl",
    show: bool = False,
):
    with open(marl_path) as f:
        m = json.load(f)

    rounds = m["rounds"]
    w = 5  # smoothing window

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        "COVER Task 3.2: MARL + Federated Learning\n"
        "Autonomous UAV Participation with DQN + Federated Q-Learning",
        fontsize=12, fontweight="bold",
    )

    # ── Accuracy ──────────────────────────────────────────────────────── #
    ax = axes[0, 0]
    ax.plot(rounds, m["accuracy"], color="steelblue", lw=1.0, alpha=0.35)
    if len(rounds) > w:
        ax.plot(rounds[w - 1:], _smooth(m["accuracy"], w),
                color="steelblue", lw=2.0, label="MARL-FL")
    else:
        ax.plot(rounds, m["accuracy"], color="steelblue", lw=2.0, label="MARL-FL")

    if fl_path and Path(fl_path).exists():
        with open(fl_path) as f:
            fl = json.load(f)
        ax.plot(fl.get("rounds", []), fl.get("accuracy", []),
                color="darkorange", lw=1.8, linestyle="--", label="Standard FL")

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Global Model Accuracy")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Mean reward ───────────────────────────────────────────────────── #
    ax = axes[0, 1]
    ax.plot(rounds, m["mean_reward"], color="seagreen", lw=1.0, alpha=0.35)
    if len(rounds) > w:
        ax.plot(rounds[w - 1:], _smooth(m["mean_reward"], w),
                color="seagreen", lw=2.0)
    ax.axhline(0, color="gray", lw=0.8, linestyle=":")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Mean Agent Reward")
    ax.set_title("DQN Reward Signal")
    ax.grid(alpha=0.3)

    # ── Participation rate ────────────────────────────────────────────── #
    ax = axes[1, 0]
    pct = [r * 100 for r in m["participation_rate"]]
    ax.plot(rounds, pct, color="mediumpurple", lw=2.0, label="MARL adaptive")
    ax.axhline(60, color="gray", lw=1.2, linestyle="--", label="Static 60%")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Participation Rate (%)")
    ax.set_title("Adaptive Participation vs Static Baseline")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Privacy budget ────────────────────────────────────────────────── #
    ax = axes[1, 1]
    ax.plot(rounds, m["privacy_epsilon"], color="tomato", lw=2.0)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Cumulative DP-ε")
    ax.set_title("Privacy Budget Consumption")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "marl_results.pdf")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"MARL results plot saved → {out}")
    if show:
        plt.show()
    plt.close()
