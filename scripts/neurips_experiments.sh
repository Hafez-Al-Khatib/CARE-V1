#!/bin/bash
set -e

echo "=========================================================="
echo " Starting NeurIPS SNN CARE Ablation Suite"
echo "=========================================================="

# Ensure environment is active before running if needed
# source /opt/mariett_team/MariettEnv/bin/activate

# 1. Setup the dataset
/opt/mariett_team/MariettEnv/bin/python scripts/setup_tiny_imagenet.py

OUT_DIR="results/v3_neurips_suite"
mkdir -p $OUT_DIR

echo "----------------------------------------------------------"
echo " Phase 1: Capacity Maximization (CIFAR-100, Normal Init)"
echo "----------------------------------------------------------"
# Target: Show that standard SG training misses capacity compared to CARE on hard datasets
/opt/mariett_team/MariettEnv/bin/python scripts/run_flexible_experiment.py --dataset cifar100 --init normal --name cifar100_control --epochs 30 --time_steps 8 --block sew --depth 18 --batch_size 64 --output_dir $OUT_DIR --no_plasticity
/opt/mariett_team/MariettEnv/bin/python scripts/run_flexible_experiment.py --dataset cifar100 --init normal --name cifar100_care --epochs 30 --time_steps 8 --block sew --depth 18 --batch_size 64 --output_dir $OUT_DIR

echo "----------------------------------------------------------"
echo " Phase 2: Ultra-Low Latency Robustness (CIFAR-10, T=2)"
echo "----------------------------------------------------------"
# Target: SNNs usually crash at T=1 and T=2. CARE dynamically boosts signals to allow learning.
/opt/mariett_team/MariettEnv/bin/python scripts/run_flexible_experiment.py --dataset cifar10 --init normal --name cifar10_T2_control --epochs 30 --time_steps 2 --block sew --depth 18 --batch_size 64 --output_dir $OUT_DIR --no_plasticity
/opt/mariett_team/MariettEnv/bin/python scripts/run_flexible_experiment.py --dataset cifar10 --init normal --name cifar10_T2_care --epochs 30 --time_steps 2 --block sew --depth 18 --batch_size 64 --output_dir $OUT_DIR

echo "----------------------------------------------------------"
echo " Phase 3: Pushing Depth on Tiny-ImageNet (ResNet-34)"
echo "----------------------------------------------------------"
# Target: Show CARE is essential for stable activations in deep residual networks natively.
/opt/mariett_team/MariettEnv/bin/python scripts/run_flexible_experiment.py --dataset tiny_imagenet --init normal --name tiny_in_res34_control --epochs 30 --time_steps 8 --block sew --depth 34 --batch_size 64 --output_dir $OUT_DIR --no_plasticity || echo "Failed to run target control"
/opt/mariett_team/MariettEnv/bin/python scripts/run_flexible_experiment.py --dataset tiny_imagenet --init normal --name tiny_in_res34_care --epochs 30 --time_steps 8 --block sew --depth 34 --batch_size 64 --output_dir $OUT_DIR || echo "Failed to run target care"


echo "=========================================================="
echo " NeurIPS Suite Complete!"
echo "=========================================================="
