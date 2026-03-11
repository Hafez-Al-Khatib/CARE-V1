"""
CARE V3 Experiment Suite - Group A: Competitive Accuracy
=========================================================
CIFAR-10 at 100 epochs with T=8 timesteps, SEW-ResNet18.
Tests Plasticity ON/OFF under Normal and Sabotage Init.

Output: results/v3/
"""

import subprocess
import sys
import time
from pathlib import Path

EXPERIMENTS = [
    {
        "name": "cifar10_norm_control_v3",
        "dataset": "cifar10",
        "init": "normal",
        "plasticity": False,
        "epochs": 100,
        "time_steps": 8,
        "eta_stdp": 0.001,
    },
    {
        "name": "cifar10_norm_care_v3",
        "dataset": "cifar10",
        "init": "normal",
        "plasticity": True,
        "epochs": 100,
        "time_steps": 8,
        "eta_stdp": 0.001,
    },
    {
        "name": "cifar10_sab_control_v3",
        "dataset": "cifar10",
        "init": "sabotage",
        "plasticity": False,
        "epochs": 100,
        "time_steps": 8,
        "eta_stdp": 0.001,
    },
    {
        "name": "cifar10_sab_care_v3",
        "dataset": "cifar10",
        "init": "sabotage",
        "plasticity": True,
        "epochs": 100,
        "time_steps": 8,
        "eta_stdp": 0.001,
    },
]


def run_experiment(exp: dict, dry_run: bool = False):
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
        "--depth", "18",
        "--batch_size", "64",
        "--output_dir", "results/v3",
    ]
    if not exp["plasticity"]:
        cmd.append("--no_plasticity")

    print(f"\n{'='*70}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*70}")

    if dry_run:
        print("  [DRY RUN] Skipping execution.")
        return

    start = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent))
    elapsed = time.time() - start
    mins = elapsed / 60

    status = "SUCCESS" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"\n  [{status}] {exp['name']}  ({mins:.1f} min)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Override output directory in run_flexible_experiment.py
    # by modifying the results path expectation
    out_base = Path("results/v3")
    out_base.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  CARE V3 - Group A: Competitive Accuracy (CIFAR-10, 100 Epochs)")
    print(f"  Total experiments: {len(EXPERIMENTS)}")
    print(f"  Output: {out_base}")
    print("=" * 70)

    total_start = time.time()
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n>>> Experiment {i}/{len(EXPERIMENTS)}: {exp['name']}")
        run_experiment(exp, dry_run=args.dry_run)

    total_mins = (time.time() - total_start) / 60
    print(f"\n{'='*70}")
    print(f"  ALL DONE! Total time: {total_mins:.1f} min ({total_mins/60:.1f} hours)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
