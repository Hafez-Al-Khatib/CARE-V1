"""
CARE V3 - Rigorous Multi-Dataset Ablation Study
===============================================
Executes rigorous ablation experiments over multiple datasets
to evaluate the effectiveness of SNR-Gated Homeostatic Plasticity (CARE). 

Datasets: Fashion-MNIST, CIFAR-10, Tiny-ImageNet
Init: Normal vs Sabotage
Plasticity: ON vs OFF
Epochs: 30
Time steps: 8
"""

import subprocess
import sys
import os
import time
from pathlib import Path

# Datasets to ablate on
DATASETS = ["fashion_mnist", "cifar10", "tiny_imagenet"]
INITS = ["normal", "sabotage"]
PLASTICITY = [True, False]

# Recalibrated Parameters for True 'Rescue'
SEVERE_SABOTAGE_STD = 0.001
TARGET_ACTIVITY_RATE = 0.02
EPOCHS = 30
TIME_STEPS = 8
ETA_STDP = 0.001

def generate_experiments():
    exps = []
    for ds in DATASETS:
        for init_mode in INITS:
            for plasticity in PLASTICITY:
                name = f"{ds}_{init_mode}_{'care' if plasticity else 'control'}_ablation"
                exp_dict = {
                    "name": name,
                    "dataset": ds,
                    "init": init_mode,
                    "plasticity": plasticity,
                    "epochs": EPOCHS, # Match CARE_Presentation.tex ablation settings
                    "time_steps": TIME_STEPS,
                    "eta_stdp": ETA_STDP,
                    "target_rate": TARGET_ACTIVITY_RATE,
                }
                if init_mode == "sabotage":
                    exp_dict["init_std"] = SEVERE_SABOTAGE_STD
                exps.append(exp_dict)
    return exps

EXPERIMENTS = generate_experiments()

def run_experiment(exp: dict, dry_run: bool = False):
    cmd = [
        sys.executable,
        "scripts/run_flexible_experiment.py",
        "--dataset", exp["dataset"],
        "--init", exp["init"],
        "--name", exp["name"],
        "--epochs", str(EPOCHS),
        "--time_steps", str(TIME_STEPS),
        "--eta_stdp", str(ETA_STDP),
        "--target_rate", str(TARGET_ACTIVITY_RATE),
        "--block", "sew",
        "--depth", "18",
        "--batch_size", "64",
        "--output_dir", "results/v3_rigorous_ablation",
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

    out_base = Path("results/v3_rigorous_ablation")
    out_base.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  CARE V3 - Rigorous Multi-Dataset Ablation Suite")
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
