"""
COVER Task 3.2 — FL + DP + Blockchain + Incentive UAV Simulation
Entry point. Run with:

  # Full simulation (all layers: FL + DP + Blockchain + Incentive + Security)
  python main.py --full

  # FL + DP only (no blockchain)
  python main.py

  # No-DP baseline comparison
  python main.py --no-dp --output results/nodp

  # Privacy–utility sweep (vary noise multiplier)
  python main.py --sweep

  # Plot from saved results
  python main.py --plot-only
"""

import argparse
import copy
import json
from pathlib import Path

import yaml

from simulation.runner import SimulationRunner
from simulation.full_runner import FullSimulationRunner
from evaluation.metrics import plot_convergence, compare_dp_vs_nodp, privacy_accuracy_tradeoff

CONFIG = "config.yaml"


def _patch_config(cfg: dict, **overrides) -> dict:
    """Return a shallow-patched copy of cfg for sweep experiments."""
    cfg = copy.deepcopy(cfg)
    for dotkey, val in overrides.items():
        keys = dotkey.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = val
    return cfg


def run_single(config_path: str = CONFIG, output_dir: str = "./results") -> dict:
    runner = SimulationRunner(config_path)
    return runner.run(output_dir=output_dir)


def run_nodp_baseline(config_path: str = CONFIG, output_dir: str = "./results/nodp") -> dict:
    """Run without DP to get the accuracy ceiling."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["dp"]["enabled"] = False

    # Temporarily write patched config
    tmp = "/tmp/cover_nodp_config.yaml"
    with open(tmp, "w") as f:
        yaml.dump(cfg, f)

    runner = SimulationRunner(tmp)
    return runner.run(output_dir=output_dir)


def run_privacy_sweep(config_path: str = CONFIG, output_dir: str = "./results/sweep"):
    """Sweep noise_multiplier to trace the privacy–accuracy Pareto curve."""
    noise_values = [0.8, 1.0, 1.1, 1.3, 1.5, 2.0]
    summaries = []
    labels = []

    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)

    for sigma in noise_values:
        cfg = _patch_config(base_cfg, **{"dp.noise_multiplier": sigma})
        tmp = f"/tmp/cover_sweep_{sigma}.yaml"
        with open(tmp, "w") as f:
            yaml.dump(cfg, f)

        sub_dir = f"{output_dir}/sigma_{sigma}"
        runner = SimulationRunner(tmp)
        summary = runner.run(output_dir=sub_dir)
        summaries.append(summary)
        labels.append(f"σ={sigma}")
        print()

    privacy_accuracy_tradeoff(summaries, labels, output_dir=output_dir, show=False)

    with open(f"{output_dir}/sweep_summary.json", "w") as f:
        json.dump({"noise_values": noise_values, "summaries": summaries}, f, indent=2)
    print(f"Sweep complete. Summary → {output_dir}/sweep_summary.json")


def main():
    parser = argparse.ArgumentParser(description="COVER Task 3.2 FL+DP UAV Simulation")
    parser.add_argument("--config", default=CONFIG, help="Path to config.yaml")
    parser.add_argument("--output", default="./results", help="Output directory")
    parser.add_argument("--no-dp", action="store_true", help="Run no-DP baseline only")
    parser.add_argument("--sweep", action="store_true", help="Run privacy–accuracy sweep")
    parser.add_argument("--compare", action="store_true",
                        help="Compare DP vs no-DP (runs both, then plots)")
    parser.add_argument("--full", action="store_true",
                        help="Run full simulation with blockchain + incentive + security")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only generate plots from existing results")
    args = parser.parse_args()

    if args.full:
        runner = FullSimulationRunner(args.config)
        runner.run(output_dir=f"{args.output}/full")
        plot_convergence(f"{args.output}/full/full_results.json", output_dir=f"{args.output}/full")
        return

    if args.plot_only:
        dp_path = f"{args.output}/convergence_results.json"
        plot_convergence(dp_path, output_dir=args.output, show=True)
        return

    if args.sweep:
        run_privacy_sweep(args.config, output_dir=f"{args.output}/sweep")
        return

    if args.no_dp:
        run_nodp_baseline(args.config, output_dir=f"{args.output}/nodp")
        return

    if args.compare:
        dp_summary = run_single(args.config, output_dir=f"{args.output}/dp")
        nodp_summary = run_nodp_baseline(args.config, output_dir=f"{args.output}/nodp")
        compare_dp_vs_nodp(
            dp_path=f"{args.output}/dp/convergence_results.json",
            nodp_path=f"{args.output}/nodp/convergence_results.json",
            output_dir=args.output,
            show=False,
        )
        return

    # Default: single DP run
    summary = run_single(args.config, output_dir=args.output)
    plot_convergence(f"{args.output}/convergence_results.json", output_dir=args.output)


if __name__ == "__main__":
    main()
