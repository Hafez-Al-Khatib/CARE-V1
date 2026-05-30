"""
CARE V3 - Dynamic Feature Starvation Stress Test
================================================
This script launches two experiments on CIFAR-10 using NORMAL initialization
but exceptionally HIGH weight decay (5e-3). 

Hypothesis: 
High weight decay will pull inactive or weakly-active weights to zero during training, 
causing catastrophic feature starvation (Loss of Capacity).
The CARE model should sense this starvation and autonomously amplify dying 
neurons, maintaining representational capacity where the Control model collapses.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_experiment(name, plasticity, weight_decay=5e-3, epochs=20):
    cmd = [
        "python3",
        "scripts/run_flexible_experiment.py",
        "--dataset", "cifar10",
        "--init", "normal",
        "--name", name,
        "--epochs", str(epochs),
        "--time_steps", "8",
        "--eta_stdp", "0.001",
        "--target_rate", "0.02",
        "--weight_decay", str(weight_decay),
        "--block", "sew",
        "--depth", "18",
        "--batch_size", "64",
        "--output_dir", "results/v3_starvation_stress_test",
    ]
    if not plasticity:
        cmd.append("--no_plasticity")

    print(f"\n{'='*70}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*70}")

    start = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent))
    elapsed = time.time() - start
    
    status = "SUCCESS" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"\n  [{status}] {name}  ({elapsed / 60:.1f} min)")


def main():
    out_base = Path("results/v3_starvation_stress_test")
    out_base.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  CARE V3 - Dynamic Feature Starvation Stress Test (CIFAR-10)")
    print("  Condition: Normal Init + Extreme Weight Decay (5e-3)")
    print("=" * 70)

    # 1. Control (No Plasticity) - Should suffer feature collapse
    run_experiment("cifar10_high_wd_control", plasticity=False)

    # 2. CARE (Plasticity ON) - Should continuously insure capacity
    run_experiment("cifar10_high_wd_care", plasticity=True)

    print("\n[SUCCESS] Starvation stress test complete.")

if __name__ == "__main__":
    main()
