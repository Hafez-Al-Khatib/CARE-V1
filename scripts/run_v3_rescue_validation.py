"""
CARE V3 - Rescue Validation
===========================
Executes a focused 30-epoch validation to prove the CARE 
Rescue phenomenon using severe sabotage initialization (std=0.001).

Tests: CIFAR-10 Sabotage (Control OFF) vs CIFAR-10 Sabotage (CARE ON)
"""

import subprocess
import sys
import os
import time
from pathlib import Path

EXPERIMENTS = [
    {
        "name": "cifar10_severe_sabotage_control",
        "dataset": "cifar10",
        "init": "sabotage",
        "plasticity": False,
        "epochs": 10,
        "time_steps": 8,
        "eta_stdp": 0.001,
    },
    {
        "name": "cifar10_severe_sabotage_care",
        "dataset": "cifar10",
        "init": "sabotage",
        "plasticity": True,
        "epochs": 10,
        "time_steps": 8,
        "eta_stdp": 0.001,
    }
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
        "--output_dir", "results/v3_rescue_validation",
    ]
    if not exp["plasticity"]:
        cmd.append("--no_plasticity")

    print(f"\n{'='*70}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*70}")

    if dry_run:
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

    out_base = Path("results/v3_rescue_validation")
    out_base.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  CARE V3 - Severe Sabotage Rescue Validation")
    print("=" * 70)

    total_start = time.time()
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n>>> Experiment {i}/{len(EXPERIMENTS)}: {exp['name']}")
        run_experiment(exp, dry_run=args.dry_run)

    total_mins = (time.time() - total_start) / 60
    print(f"\n{'='*70}")
    print(f"  ALL DONE! Total time: {total_mins:.1f} min")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
