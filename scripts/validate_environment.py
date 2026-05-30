#!/usr/bin/env python3
"""
CARE Environment Validation
============================
Checks that the remote RTX 4090 / Linux machine is ready to run CARE experiments.
Run this before launching any suite to catch missing deps, CUDA issues, or data gaps.

Usage:
    python scripts/validate_environment.py
    python scripts/validate_environment.py --check-data
"""

import argparse
import importlib
import sys
from pathlib import Path


def check_module(name: str, import_name: str = None) -> bool:
    """Check if a Python module is installed."""
    try:
        importlib.import_module(import_name or name)
        print(f"  [OK]   {name}")
        return True
    except ImportError:
        print(f"  [FAIL] {name} — install with pip")
        return False


def check_cuda() -> bool:
    """Check PyTorch CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  [OK]   CUDA available: {device} ({mem:.1f} GB)")
            return True
        else:
            print(f"  [WARN] CUDA not available — will run on CPU (very slow)")
            return False
    except Exception as e:
        print(f"  [FAIL] CUDA check error: {e}")
        return False


def check_datasets(check_data: bool = False) -> bool:
    """Check if required datasets are available or downloadable."""
    data_dir = Path("data")
    ok = True

    # Fashion-MNIST is auto-downloadable
    print(f"  [OK]   Fashion-MNIST (auto-download)")

    # CIFAR-10/100 are auto-downloadable
    print(f"  [OK]   CIFAR-10 (auto-download)")
    print(f"  [OK]   CIFAR-100 (auto-download)")

    # Tiny-ImageNet requires manual download
    tiny_path = data_dir / "tiny-imagenet-200"
    if tiny_path.exists():
        print(f"  [OK]   Tiny-ImageNet found at {tiny_path}")
    else:
        print(f"  [WARN] Tiny-ImageNet NOT found at {tiny_path}")
        print(f"         Download from: http://cs231n.stanford.edu/tiny-imagenet-200.zip")
        if check_data:
            ok = False

    return ok


def check_code_integrity() -> bool:
    """Check that critical source files exist and import correctly."""
    print("\n[4/5] Code Integrity")
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    checks = [
        ("models.components.neuron", "CareResNet, CareLIFConv"),
        ("systems.experiment", "DeadNeuronExperiment"),
        ("systems.modern_experiment", "ModernArchExperiment"),
        ("scripts.run_flexible_experiment", "Experiment runner"),
    ]

    ok = True
    for module_name, desc in checks:
        try:
            importlib.import_module(module_name)
            print(f"  [OK]   {module_name} ({desc})")
        except Exception as e:
            print(f"  [FAIL] {module_name} — {e}")
            ok = False

    # Run smoke test if available
    smoke_path = project_root / "scripts" / "smoke_test.py"
    if smoke_path.exists():
        print(f"  [INFO] Running smoke_test.py...")
        import subprocess
        result = subprocess.run(
            [sys.executable, str(smoke_path)],
            capture_output=True, text=True, cwd=str(project_root)
        )
        if result.returncode == 0:
            print(f"  [OK]   smoke_test.py passed")
        else:
            print(f"  [FAIL] smoke_test.py failed:\n{result.stderr}")
            ok = False

    return ok


def check_disk_space() -> bool:
    """Check available disk space."""
    import shutil
    stat = shutil.disk_usage(".")
    free_gb = stat.free / (1024**3)
    print(f"\n[5/5] Disk Space")
    print(f"  Free: {free_gb:.1f} GB")
    if free_gb < 10:
        print(f"  [WARN] Less than 10 GB free — may run out of space for checkpoints")
        return False
    else:
        print(f"  [OK]   Sufficient disk space")
        return True


def main():
    parser = argparse.ArgumentParser(description="CARE Environment Validation")
    parser.add_argument("--check-data", action="store_true",
                        help="Fail if Tiny-ImageNet is missing")
    args = parser.parse_args()

    print("=" * 60)
    print("CARE ENVIRONMENT VALIDATION")
    print("=" * 60)

    print("\n[1/5] Python Packages")
    deps_ok = True
    deps_ok &= check_module("torch")
    deps_ok &= check_module("pytorch_lightning", "pytorch_lightning")
    deps_ok &= check_module("snntorch")
    deps_ok &= check_module("torchvision")
    deps_ok &= check_module("pandas")
    deps_ok &= check_module("numpy")
    deps_ok &= check_module("tqdm")
    deps_ok &= check_module("hydra-core", "hydra")
    check_module("tonic")  # Optional, for DVS datasets
    check_module("wandb")  # Optional

    print("\n[2/5] CUDA / GPU")
    cuda_ok = check_cuda()

    print("\n[3/5] Datasets")
    data_ok = check_datasets(check_data=args.check_data)

    code_ok = check_code_integrity()
    disk_ok = check_disk_space()

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Dependencies: {'PASS' if deps_ok else 'FAIL'}")
    print(f"  CUDA/GPU:     {'PASS' if cuda_ok else 'WARN (CPU)'}")
    print(f"  Datasets:     {'PASS' if data_ok else 'FAIL'}")
    print(f"  Code:         {'PASS' if code_ok else 'FAIL'}")
    print(f"  Disk:         {'PASS' if disk_ok else 'WARN'}")

    all_ok = deps_ok and data_ok and code_ok
    if all_ok:
        print("\n  ✅ Environment is ready for CARE experiments!")
        return 0
    else:
        print("\n  ❌ Please fix the issues above before running experiments.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
