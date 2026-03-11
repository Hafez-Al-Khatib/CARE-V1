"""
CARE V3 Experiment Suite - Combined Orchestrator
=================================================
Runs all experiment groups sequentially:
  Group A: CIFAR-10 competitive accuracy (BN-fixed, base_channels=32)
  Group B: η_stdp sensitivity sweep (FMNIST, sabotage)
  Group C: Depth scaling (FMNIST, sabotage, depth 6/18/34)

Usage:
  python scripts/run_all_experiments.py                    # Run all groups
  python scripts/run_all_experiments.py --group A          # Run only Group A
  python scripts/run_all_experiments.py --group B          # Run only Group B
  python scripts/run_all_experiments.py --group C          # Run only Group C
  python scripts/run_all_experiments.py --dry-run          # Print commands only
  python scripts/run_all_experiments.py --group A --dry-run

Output: results/v3/
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path


# ============================================================================
# Group A: Competitive Accuracy on CIFAR-10
# Architecture now fixed: BN applied, base_channels=32, 3x3 stem
# ============================================================================
GROUP_A = [
    {
        "name": "cifar10_norm_control_v3",
        "dataset": "cifar10",
        "init": "normal",
        "plasticity": False,
        "epochs": 100,
        "time_steps": 8,
        "eta_stdp": 0.001,
        "base_channels": 32,
        "lr": 1e-3,
        "seed": 42,
    },
    {
        "name": "cifar10_norm_care_v3",
        "dataset": "cifar10",
        "init": "normal",
        "plasticity": True,
        "epochs": 100,
        "time_steps": 8,
        "eta_stdp": 0.001,
        "base_channels": 32,
        "lr": 1e-3,
        "seed": 42,
    },
    {
        "name": "cifar10_sab_control_v3",
        "dataset": "cifar10",
        "init": "sabotage",
        "plasticity": False,
        "epochs": 100,
        "time_steps": 8,
        "eta_stdp": 0.001,
        "base_channels": 32,
        "lr": 1e-3,
        "seed": 42,
    },
    {
        "name": "cifar10_sab_care_v3",
        "dataset": "cifar10",
        "init": "sabotage",
        "plasticity": True,
        "epochs": 100,
        "time_steps": 8,
        "eta_stdp": 0.001,
        "base_channels": 32,
        "lr": 1e-3,
        "seed": 42,
    },
]


# ============================================================================
# Group B: η_stdp Sensitivity Sweep (FMNIST, Sabotage)
# ============================================================================
GROUP_B = [
    {
        "name": f"fmnist_sab_eta{str(eta).replace('.', 'p')}",
        "dataset": "fashion_mnist",
        "init": "sabotage",
        "plasticity": True,
        "epochs": 30,
        "time_steps": 8,
        "eta_stdp": eta,
        "base_channels": 32,
        "lr": 1e-3,
        "seed": 42,
    }
    for eta in [0.0001, 0.0005, 0.001, 0.005, 0.01]
]


# ============================================================================
# Group C: Depth Scaling (FMNIST, Sabotage, depth 6/18/34)
# ============================================================================
GROUP_C = []
for depth in [6, 18, 34]:
    GROUP_C.append({
        "name": f"fmnist_sab_d{depth}_control",
        "dataset": "fashion_mnist",
        "init": "sabotage",
        "plasticity": False,
        "epochs": 30,
        "time_steps": 8,
        "eta_stdp": 0.001,
        "base_channels": 32,
        "lr": 1e-3,
        "seed": 42,
        "depth": depth,
    })
    GROUP_C.append({
        "name": f"fmnist_sab_d{depth}_care",
        "dataset": "fashion_mnist",
        "init": "sabotage",
        "plasticity": True,
        "epochs": 30,
        "time_steps": 8,
        "eta_stdp": 0.001,
        "base_channels": 32,
        "lr": 1e-3,
        "seed": 42,
        "depth": depth,
    })


ALL_GROUPS = {
    "A": ("Competitive Accuracy (CIFAR-10, 100 epochs)", GROUP_A),
    "B": ("η_stdp Sensitivity Sweep (FMNIST)", GROUP_B),
    "C": ("Depth Scaling (FMNIST, d=6/18/34)", GROUP_C),
}


def build_cmd(exp: dict) -> list:
    cmd = [
        sys.executable,
        "scripts/run_flexible_experiment.py",
        "--dataset", exp["dataset"],
        "--init", exp["init"],
        "--name", exp["name"],
        "--epochs", str(exp["epochs"]),
        "--time_steps", str(exp["time_steps"]),
        "--eta_stdp", str(exp["eta_stdp"]),
        "--block", "sew",
        "--depth", str(exp.get("depth", 18)),
        "--batch_size", "64",
        "--output_dir", "results/v3",
        "--base_channels", str(exp["base_channels"]),
        "--seed", str(exp["seed"]),
        "--lr", str(exp["lr"]),
        "--num_workers", "2",
    ]
    if not exp["plasticity"]:
        cmd.append("--no_plasticity")
    return cmd


def run_experiment(exp: dict, idx: int, total: int, dry_run: bool = False):
    cmd = build_cmd(exp)

    print(f"\n{'='*70}")
    print(f"  [{idx}/{total}] {exp['name']}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*70}")

    if dry_run:
        print("  [DRY RUN] Skipping execution.")
        return 0

    start = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent))
    elapsed = time.time() - start
    mins = elapsed / 60

    status = "SUCCESS" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"\n  [{status}] {exp['name']}  ({mins:.1f} min)")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="CARE V3 Combined Orchestrator")
    parser.add_argument("--group", type=str, default=None, choices=["A", "B", "C"],
                        help="Run only a specific group (default: all)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Select groups
    if args.group:
        groups_to_run = {args.group: ALL_GROUPS[args.group]}
    else:
        groups_to_run = ALL_GROUPS

    # Count total experiments
    total_exps = sum(len(exps) for _, exps in groups_to_run.values())

    print("=" * 70)
    print("  CARE V3 - Publication-Grade Experiment Suite")
    print(f"  Groups: {', '.join(groups_to_run.keys())}")
    print(f"  Total experiments: {total_exps}")
    print(f"  Output: results/v3/")
    print("=" * 70)

    total_start = time.time()
    exp_idx = 0
    failures = []

    for group_key, (group_desc, experiments) in groups_to_run.items():
        print(f"\n{'#'*70}")
        print(f"  GROUP {group_key}: {group_desc}")
        print(f"  Experiments in group: {len(experiments)}")
        print(f"{'#'*70}")

        for exp in experiments:
            exp_idx += 1
            rc = run_experiment(exp, exp_idx, total_exps, dry_run=args.dry_run)
            if rc != 0:
                failures.append(exp["name"])

    total_mins = (time.time() - total_start) / 60
    print(f"\n{'='*70}")
    print(f"  ALL DONE! Total time: {total_mins:.1f} min ({total_mins/60:.1f} hours)")
    if failures:
        print(f"  FAILURES: {failures}")
    else:
        print(f"  All {total_exps} experiments completed successfully.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
