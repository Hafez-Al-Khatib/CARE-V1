#!/usr/bin/env python3
"""
Checkpoint Cleanup Utility
==========================
Removes intermediate checkpoints to save disk space, keeping only:
- The best checkpoint (highest val/accuracy)
- Optionally the final checkpoint

Usage:
    # Clean a specific experiment directory
    python scripts/checkpoint_cleanup.py --experiment-dir results/remote/my_exp

    # Clean all experiment directories under results/
    python scripts/checkpoint_cleanup.py --results-root results --dry-run

    # Keep best + final, delete everything else
    python scripts/checkpoint_cleanup.py --experiment-dir results/remote/my_exp --keep-final
"""

import argparse
import shutil
from pathlib import Path


def clean_experiment_dir(exp_dir: Path, keep_final: bool = False, dry_run: bool = False) -> dict:
    """Clean checkpoints in a single experiment directory."""
    checkpoint_dir = exp_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return {"status": "no_checkpoints", "deleted": 0, "kept": 0}

    ckpt_files = sorted(checkpoint_dir.glob("*.ckpt"))
    if not ckpt_files:
        return {"status": "empty", "deleted": 0, "kept": 0}

    # Identify best checkpoint by filename (contains val/accuracy)
    best_ckpt = None
    best_score = -1.0
    final_ckpt = None
    final_epoch = -1

    for ckpt in ckpt_files:
        name = ckpt.stem
        # Try to extract val_accuracy from filename like "best-epoch=05-val/accuracy=0.7234"
        if "val" in name and "=" in name:
            try:
                score_part = name.split("=")[-1]
                score = float(score_part)
                if score > best_score:
                    best_score = score
                    best_ckpt = ckpt
            except ValueError:
                pass

        # Try to extract epoch number
        if "epoch=" in name:
            try:
                epoch_part = name.split("epoch=")[1].split("-")[0]
                epoch = int(epoch_part)
                if epoch > final_epoch:
                    final_epoch = epoch
                    final_ckpt = ckpt
            except (ValueError, IndexError):
                pass

    # Fallback: if no score found, keep the most recently modified
    if best_ckpt is None:
        best_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)

    to_keep = {best_ckpt}
    if keep_final and final_ckpt is not None and final_ckpt != best_ckpt:
        to_keep.add(final_ckpt)

    deleted = 0
    kept = 0
    for ckpt in ckpt_files:
        if ckpt in to_keep:
            kept += 1
            print(f"  [KEEP] {ckpt.name}")
        else:
            deleted += 1
            if dry_run:
                print(f"  [DEL*] {ckpt.name} (dry-run)")
            else:
                ckpt.unlink()
                print(f"  [DEL]  {ckpt.name}")

    return {
        "status": "cleaned",
        "deleted": deleted,
        "kept": kept,
        "best": best_ckpt.name if best_ckpt else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Checkpoint Cleanup")
    parser.add_argument("--experiment-dir", type=str, default=None,
                        help="Single experiment directory to clean")
    parser.add_argument("--results-root", type=str, default=None,
                        help="Root results directory to clean recursively")
    parser.add_argument("--keep-final", action="store_true",
                        help="Also keep the final epoch checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without removing")
    args = parser.parse_args()

    total_deleted = 0
    total_kept = 0

    if args.experiment_dir:
        result = clean_experiment_dir(
            Path(args.experiment_dir),
            keep_final=args.keep_final,
            dry_run=args.dry_run,
        )
        total_deleted += result["deleted"]
        total_kept += result["kept"]

    elif args.results_root:
        root = Path(args.results_root)
        if not root.exists():
            print(f"[ERROR] Directory not found: {root}")
            return

        # Find all directories containing checkpoints
        for checkpoint_dir in root.rglob("checkpoints"):
            exp_dir = checkpoint_dir.parent
            print(f"\n[CLEANING] {exp_dir}")
            result = clean_experiment_dir(
                exp_dir,
                keep_final=args.keep_final,
                dry_run=args.dry_run,
            )
            total_deleted += result["deleted"]
            total_kept += result["kept"]

    else:
        print("[ERROR] Specify either --experiment-dir or --results-root")
        return

    print(f"\n{'='*60}")
    print(f"CHECKPOINT CLEANUP SUMMARY")
    print(f"{'='*60}")
    print(f"Deleted: {total_deleted}")
    print(f"Kept:    {total_kept}")
    print(f"Space saved: ~{total_deleted * 50:.0f} MB (est. 50 MB per ckpt)")


if __name__ == "__main__":
    main()
