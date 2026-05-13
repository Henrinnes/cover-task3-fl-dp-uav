"""
Visualisation and analysis utilities for FL + DP simulation results.
All plots are saved as PDF for publication quality.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.8,
})


def _load(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def plot_convergence(results_path: str, output_dir: str = "./results", show: bool = False):
    """Four-panel plot: accuracy, loss, privacy budget, participation."""
    data = _load(results_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rounds = data["rounds"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("COVER Task 3.2 — FL + DP in UAV Network", fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(rounds, data["accuracies"], "b-o", markersize=3, label="Test accuracy")
    ax.axhline(data["target_accuracy"], color="r", ls="--", lw=1.2, label=f"Target {data['target_accuracy']}")
    if data["convergence_round"]:
        ax.axvline(data["convergence_round"], color="g", ls=":", lw=1.2,
                   label=f"Converge @ round {data['convergence_round']}")
    ax.set_xlabel("Round"); ax.set_ylabel("Accuracy"); ax.set_title("Test Accuracy"); ax.legend()

    ax = axes[0, 1]
    ax.plot(rounds, data["losses"], "r-o", markersize=3, color="tomato")
    ax.set_xlabel("Round"); ax.set_ylabel("Cross-entropy loss"); ax.set_title("Test Loss")

    ax = axes[1, 0]
    ax.plot(rounds, data["epsilons"], "g-o", markersize=3, color="seagreen")
    ax.set_xlabel("Round"); ax.set_ylabel("ε (privacy budget spent)"); ax.set_title("Cumulative ε (RDP accountant)")

    ax = axes[1, 1]
    ax.bar(rounds, data["participant_counts"], color="steelblue", alpha=0.7, width=max(rounds)/len(rounds)*0.8)
    ax.set_xlabel("Round"); ax.set_ylabel("# Participating UAVs"); ax.set_title("UAV Participation per Round")

    plt.tight_layout()
    out = f"{output_dir}/fl_dp_convergence.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    if show:
        plt.show()
    return fig


def compare_dp_vs_nodp(
    dp_path: str, nodp_path: str, output_dir: str = "./results", show: bool = False
):
    """Overlay accuracy curves: DP vs. no-DP to quantify the privacy-utility gap."""
    dp = _load(dp_path)
    nodp = _load(nodp_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dp["rounds"], dp["accuracies"], "b-o", markersize=3, label="FL + DP-SGD")
    ax.plot(nodp["rounds"], nodp["accuracies"], "r--s", markersize=3, label="FL (no DP)")
    ax.axhline(dp["target_accuracy"], color="k", ls=":", lw=1.0, label=f"Target {dp['target_accuracy']}")
    ax.set_xlabel("Communication round"); ax.set_ylabel("Test accuracy")
    ax.set_title("Privacy–Utility Tradeoff: DP vs. no-DP FL in UAV Network")
    ax.legend()
    plt.tight_layout()
    out = f"{output_dir}/dp_vs_nodp.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    if show:
        plt.show()
    return fig


def privacy_accuracy_tradeoff(
    results: List[Dict],
    labels: List[str],
    output_dir: str = "./results",
    show: bool = False,
):
    """
    Plot final accuracy vs. final ε for multiple runs with different noise_multiplier.
    Useful for the privacy–accuracy Pareto curve in the paper.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    epsilons = [r["final_epsilon"] for r in results]
    accuracies = [r["final_accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epsilons, accuracies, "o-", color="steelblue", markersize=7)
    for eps, acc, lbl in zip(epsilons, accuracies, labels):
        ax.annotate(lbl, (eps, acc), textcoords="offset points", xytext=(4, 4), fontsize=9)
    ax.set_xlabel("Privacy budget ε"); ax.set_ylabel("Final test accuracy")
    ax.set_title("Privacy–Accuracy Pareto Curve (COVER Task 3.2)")
    plt.tight_layout()
    out = f"{output_dir}/privacy_accuracy_pareto.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    if show:
        plt.show()
    return fig
