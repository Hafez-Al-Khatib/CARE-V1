#!/usr/bin/env python3
"""
CARE Remote Experiment Runner
==============================
Launches CARE experiments inside persistent tmux sessions with full logging,
reproducibility seeds, and automatic checkpoint cleanup.

Designed for SSH-based remote execution on RTX 4090 / A100 servers where
connections may drop.

Usage:
    # Launch a single experiment in a tmux session
    python scripts/remote_runner.py --name plain8_cifar10_care \
        --dataset cifar10 --arch plain --depth 8 --epochs 100

    # Launch a full ablation suite
    python scripts/remote_runner.py --suite ablation_plain --dry-run

    # Attach to a running tmux session
    tmux attach -t care_plain8_cifar10_care

    # List all CARE tmux sessions
    tmux ls | grep care_

Requirements:
    - tmux installed on the remote server
    - Python environment with pytorch, lightning, snntorch
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_TMUX_PREFIX = "care_"
DEFAULT_LOG_DIR = "logs/remote"
DEFAULT_CHECKPOINT_DIR = "checkpoints/remote"


# ============================================================================
# Experiment Registry
# ============================================================================

SUITES = {
    "lazarus_vgg8": [
        {
            "name": "vgg8_cifar10_control",
            "dataset": "cifar10",
            "arch": "vgg",
            "depth": 8,
            "epochs": 100,
            "no_plasticity": True,
        },
        {
            "name": "vgg8_cifar10_care",
            "dataset": "cifar10",
            "arch": "vgg",
            "depth": 8,
            "epochs": 100,
            "no_plasticity": False,
        },
    ],
    "lazarus_plain": [
        {
            "name": "plain8_cifar10_control",
            "dataset": "cifar10",
            "arch": "plain",
            "depth": 8,
            "epochs": 100,
            "no_plasticity": True,
        },
        {
            "name": "plain8_cifar10_care",
            "dataset": "cifar10",
            "arch": "plain",
            "depth": 8,
            "epochs": 100,
            "no_plasticity": False,
        },
        {
            "name": "plain18_cifar10_control",
            "dataset": "cifar10",
            "arch": "plain",
            "depth": 18,
            "epochs": 100,
            "no_plasticity": True,
        },
        {
            "name": "plain18_cifar10_care",
            "dataset": "cifar10",
            "arch": "plain",
            "depth": 18,
            "epochs": 100,
            "no_plasticity": False,
        },
    ],
    "ablation_plain": [
        {"name": "plain8_care_gamma", "dataset": "cifar10", "arch": "plain", "depth": 8, "homeo_target": "gamma", "epochs": 50},
        {"name": "plain8_care_weight", "dataset": "cifar10", "arch": "plain", "depth": 8, "homeo_target": "weight", "epochs": 50},
        {"name": "plain8_care_both", "dataset": "cifar10", "arch": "plain", "depth": 8, "homeo_target": "both", "epochs": 50},
        {"name": "plain8_care_snr_off", "dataset": "cifar10", "arch": "plain", "depth": 8, "snr_off": True, "epochs": 50},
        {"name": "plain8_care_snr_on", "dataset": "cifar10", "arch": "plain", "depth": 8, "snr_off": False, "epochs": 50},
        {"name": "plain8_care_eta1e4", "dataset": "cifar10", "arch": "plain", "depth": 8, "eta_stdp": 0.0001, "epochs": 50},
        {"name": "plain8_care_eta1e3", "dataset": "cifar10", "arch": "plain", "depth": 8, "eta_stdp": 0.001, "epochs": 50},
        {"name": "plain8_care_eta1e2", "dataset": "cifar10", "arch": "plain", "depth": 8, "eta_stdp": 0.01, "epochs": 50},
    ],
    "baseline_resnet": [
        {
            "name": "resnet18_cifar10_control",
            "dataset": "cifar10",
            "arch": "resnet",
            "depth": 18,
            "block": "sew",
            "epochs": 100,
            "no_plasticity": True,
        },
        {
            "name": "resnet18_cifar10_care",
            "dataset": "cifar10",
            "arch": "resnet",
            "depth": 18,
            "block": "sew",
            "epochs": 100,
            "no_plasticity": False,
        },
    ],
}


def build_cmd(exp: dict) -> list:
    """Build the CLI command for run_flexible_experiment.py."""
    cmd = [
        sys.executable,
        "scripts/run_flexible_experiment.py",
        "--dataset", exp["dataset"],
        "--arch", exp.get("arch", "resnet"),
        "--depth", str(exp.get("depth", 18)),
        "--name", exp["name"],
        "--epochs", str(exp["epochs"]),
        "--time_steps", str(exp.get("time_steps", 16)),
        "--eta_stdp", str(exp.get("eta_stdp", 0.005)),
        "--target_rate", str(exp.get("target_rate", 0.02)),
        "--lr", str(exp.get("lr", 1e-3)),
        "--batch_size", str(exp.get("batch_size", 64)),
        "--base_channels", str(exp.get("base_channels", 32)),
        "--seed", str(exp.get("seed", 42)),
        "--output_dir", exp.get("output_dir", "results/remote"),
        "--homeo_target", exp.get("homeo_target", "gamma"),
        "--num_workers", str(exp.get("num_workers", 4)),
    ]
    if exp.get("no_plasticity"):
        cmd.append("--no_plasticity")
    if exp.get("snr_off"):
        cmd.append("--snr_off")
    if exp.get("init"):
        cmd.extend(["--init", exp["init"]])
    if exp.get("init_std") is not None:
        cmd.extend(["--init_std", str(exp["init_std"])])
    if exp.get("block"):
        cmd.extend(["--block", exp["block"]])
    return cmd


def make_tmux_command(session_name: str, log_file: Path, cmd: list) -> list:
    """Build a tmux command that runs the experiment and logs everything."""
    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Command that runs inside tmux:
    # 1. Print header
    # 2. Run experiment, tee output to log
    # 3. Run checkpoint cleanup after
    # 4. Print footer
    # Pre-compute experiment directory for cleanup
    try:
        out_idx = cmd.index("--output_dir") + 1
        name_idx = cmd.index("--name") + 1
        exp_dir = str(Path(cmd[out_idx]) / cmd[name_idx])
    except (ValueError, IndexError):
        exp_dir = ""

    cmd_str = " ".join(cmd)

    inner = (
        f'echo "=== CARE Remote Run: {session_name} ===" && '
        f'echo "Started: $(date)" && '
        f'echo "CUDA: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" && '
        f'echo "Command: {cmd_str}" && '
        f'echo "========================================" && '
        f'{cmd_str} 2>&1 | tee "{log_file}" && '
        f'python scripts/checkpoint_cleanup.py --experiment-dir "{exp_dir}" && '
        f'echo "" && '
        f'echo "=== Run Complete ===" && '
        f'echo "Finished: $(date)" && '
        f'echo "Log: {log_file}"'
    )

    return [
        "tmux", "new-session", "-d", "-s", session_name,
        "bash", "-c", inner,
    ]


def session_exists(name: str) -> bool:
    """Check if a tmux session already exists."""
    result = subprocess.run(
        ["tmux", "has-session", "-t", name],
        capture_output=True,
    )
    return result.returncode == 0


def run_single(exp: dict, dry_run: bool = False, force: bool = False) -> str:
    """Launch a single experiment in a tmux session."""
    session_name = f"{DEFAULT_TMUX_PREFIX}{exp['name']}"
    log_file = Path(DEFAULT_LOG_DIR) / f"{exp['name']}.log"

    if session_exists(session_name) and not force:
        print(f"[SKIP] tmux session '{session_name}' already exists. Use --force to overwrite.")
        return session_name

    if session_exists(session_name) and force:
        print(f"[KILL] Removing existing session '{session_name}'")
        if not dry_run:
            subprocess.run(["tmux", "kill-session", "-t", session_name], capture_output=True)

    cmd = build_cmd(exp)
    tmux_cmd = make_tmux_command(session_name, log_file, cmd)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp['name']}")
    print(f"TMUX SESSION: {session_name}")
    print(f"LOG FILE: {log_file}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*70}")

    if dry_run:
        print("[DRY RUN] Would execute:")
        print(" ".join(tmux_cmd))
        return session_name

    result = subprocess.run(tmux_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Failed to create tmux session: {result.stderr}")
    else:
        print(f"[LAUNCHED] tmux session '{session_name}' is running.")
        print(f"[ATTACH]   tmux attach -t {session_name}")
        print(f"[LOG]      tail -f {log_file}")

    return session_name


def main():
    parser = argparse.ArgumentParser(description="CARE Remote Experiment Runner")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--arch", type=str, default="resnet")
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--block", type=str, default="sew")
    parser.add_argument("--suite", type=str, default=None, choices=list(SUITES.keys()),
                        help="Run a predefined experiment suite")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--force", action="store_true", help="Kill existing tmux sessions with same name")
    parser.add_argument("--no-plasticity", action="store_true", dest="no_plasticity", help="Disable plasticity")
    parser.add_argument("--homeo-target", type=str, default="gamma")
    parser.add_argument("--eta-stdp", type=float, default=0.005)
    parser.add_argument("--target-rate", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--time-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--snr-off", action="store_true")
    args = parser.parse_args()

    # Determine experiments to run
    if args.suite:
        experiments = SUITES[args.suite]
        print(f"[SUITE] Running '{args.suite}' with {len(experiments)} experiments")
    elif args.name:
        experiments = [{
            "name": args.name,
            "dataset": args.dataset,
            "arch": args.arch,
            "depth": args.depth,
            "epochs": args.epochs,
            "block": args.block,
            "no_plasticity": args.no_plasticity,
            "homeo_target": args.homeo_target,
            "eta_stdp": args.eta_stdp,
            "target_rate": args.target_rate,
            "batch_size": args.batch_size,
            "base_channels": args.base_channels,
            "time_steps": args.time_steps,
            "lr": args.lr,
            "seed": args.seed,
            "num_workers": args.num_workers,
            "snr_off": args.snr_off,
        }]
    else:
        parser.error("Either --suite or --name must be provided.")

    # Launch all experiments
    sessions = []
    for exp in experiments:
        sessions.append(run_single(exp, dry_run=args.dry_run, force=args.force))

    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(sessions)} experiment(s) scheduled")
    print(f"{'='*70}")
    for s in sessions:
        print(f"  tmux attach -t {s}")
    print(f"\nTo monitor all logs:")
    print(f"  tail -f logs/remote/*.log")
    print(f"\nTo list all CARE sessions:")
    print(f"  tmux ls | grep {DEFAULT_TMUX_PREFIX}")


if __name__ == "__main__":
    main()
