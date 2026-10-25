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
    fig.suptitle("COVER Task 3.2: FL + DP in UAV Network", fontsize=13, fontweight="bold")

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


def plot_full_results(
    fl_path: str,
    bc_path: str,
    output_dir: str = "./results/full",
    show: bool = False,
) -> str:
    """
    Six-panel figure for the full simulation (all layers active).
      Row 1: Test accuracy | Cumulative DP-ε | Participation per round
      Row 2: Stackelberg λ* | Blocked updates per round | Token leaderboard
    """
    fl = _load(fl_path)
    bc = _load(bc_path)

    rounds     = fl["rounds"]
    accuracies = fl["accuracies"]
    epsilons   = fl["epsilons"]
    n_part     = fl["participant_counts"]
    lambdas    = bc["lambda_stars"]
    blocked    = bc["blocked_counts"]
    leaderboard = bc.get("leaderboard", [])
    balances   = bc.get("final_balances", {})

    bc_rounds = list(range(1, len(lambdas) + 1))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "COVER Task 3.2 Full Simulation: FL + DP + Blockchain + Incentive + Security",
        fontsize=12, fontweight="bold",
    )

    # ── Accuracy ──────────────────────────────────────────────────────── #
    ax = axes[0, 0]
    ax.plot(rounds, accuracies, "b-o", markersize=4, label="DP-FL accuracy")
    if fl.get("target_accuracy"):
        ax.axhline(fl["target_accuracy"], color="r", ls="--", lw=1.2,
                   label=f"Target {fl['target_accuracy']}")
    if fl.get("convergence_round"):
        ax.axvline(fl["convergence_round"], color="g", ls=":", lw=1.2,
                   label=f"Converge @ {fl['convergence_round']}")
    ax.set_xlabel("Round"); ax.set_ylabel("Test Accuracy")
    ax.set_title("Global Model Accuracy"); ax.legend(fontsize=8)

    # ── DP-ε ──────────────────────────────────────────────────────────── #
    ax = axes[0, 1]
    ax.plot(rounds, epsilons, color="seagreen", marker="o", markersize=4)
    ax.set_xlabel("Round"); ax.set_ylabel("Cumulative DP-ε")
    ax.set_title("Privacy Budget (RDP Accountant)")

    # ── Participation ─────────────────────────────────────────────────── #
    ax = axes[0, 2]
    w = max(rounds) / len(rounds) * 0.8
    ax.bar(rounds, n_part, color="steelblue", alpha=0.75, width=w)
    ax.set_xlabel("Round"); ax.set_ylabel("# Participating UAVs")
    ax.set_title("Channel-Aware Participation")

    # ── Stackelberg λ* ───────────────────────────────────────────────── #
    ax = axes[1, 0]
    ax.plot(bc_rounds, lambdas, color="darkorange", lw=1.8)
    ax.set_xlabel("Round"); ax.set_ylabel("Optimal reward rate λ*")
    ax.set_title("Stackelberg Equilibrium λ*")

    # ── Blocked updates ──────────────────────────────────────────────── #
    ax = axes[1, 1]
    w2 = max(bc_rounds) / len(bc_rounds) * 0.8
    ax.bar(bc_rounds, blocked, color="tomato", alpha=0.8, width=w2)
    ax.set_xlabel("Round"); ax.set_ylabel("Updates blocked")
    ax.set_title("Byzantine + Free-rider Detections")

    # ── Token leaderboard ─────────────────────────────────────────────── #
    ax = axes[1, 2]
    attacker_ids = set(bc.get("attacker_ids", []))
    if leaderboard:
        # leaderboard entries are (uav_id, balance) pairs — 2-element lists
        # once round-tripped through JSON.
        top = leaderboard[:8]
        ids  = [f"UAV {e[0]}" for e in top]
        bals = [e[1] for e in top]
        cols = ["gold" if e[0] in attacker_ids else "steelblue" for e in top]
    elif balances:
        top  = sorted(balances.items(), key=lambda x: x[1], reverse=True)[:8]
        ids  = [f"UAV {k}" for k, _ in top]
        bals = [v for _, v in top]
        cols = ["gold" if k in attacker_ids else "steelblue" for k, _ in top]
    else:
        ids, bals, cols = [], [], []

    if ids:
        ax.barh(ids, bals, color=cols, alpha=0.85)
        ax.invert_yaxis()
        ax.set_xlabel("Token Balance")
    ax.set_title("Top-8 UAV Token Leaderboard")

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out = f"{output_dir}/full_simulation_results.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    if show:
        plt.show()
    plt.close()
    return out
