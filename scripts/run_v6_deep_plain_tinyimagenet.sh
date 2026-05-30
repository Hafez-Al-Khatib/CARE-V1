#!/bin/bash

# ==============================================================================
# CARE V6: Deep Scale Pipeline - "The Neuromorphic Hardware Argument"
# ==============================================================================
# This script executes the deep Plain-34 experiments on Tiny-ImageNet.
# The goal is to prove that CARE is computationally essential for catastrophic-
# collapse environments where skip connections (ResNets) are unavailable.
# ==============================================================================

set -e

RESULTS_DIR="results/v6_deep_plain_tinyimagenet"
EPOCHS=30
BS=16
LR=1e-3
T=8
DEPTH=34
ARCH="plain"

# Create log directory
mkdir -p "$RESULTS_DIR"
echo "Starting V6 Deep Scale Tiny-ImageNet Pipeline log..." > "$RESULTS_DIR/master.log"

# --- EXPERIMENT 1: Control (No Plasticity) ---
# ALREADY FINISHED & SAVED. Commenting out to save 8+ hours.
# echo "========================================================="
# echo "Running Control Plain-$DEPTH on Tiny-ImageNet..."
# echo "========================================================="
# python3 scripts/run_flexible_experiment.py \
#     --dataset tiny_imagenet \
#     --arch "$ARCH" \
#     --depth "$DEPTH" \
#     --epochs "$EPOCHS" \
#     --batch_size "$BS" \
#     --lr "$LR" \
#     --time_steps "$T" \
#     --no_plasticity \
#     --name "Control_Plain${DEPTH}" \
#     --output_dir "$RESULTS_DIR" \
#     --block sew # ignored for plain arch

# --- EXPERIMENT 2: CARE (Homeostatic Weight Scaling) ---
echo "========================================================="
echo "Running CARE Plain-$DEPTH on Tiny-ImageNet (Strong Dose 0.01)..."
echo "========================================================="
python3 scripts/run_flexible_experiment.py \
    --dataset tiny_imagenet \
    --arch "$ARCH" \
    --depth "$DEPTH" \
    --epochs "$EPOCHS" \
    --batch_size "$BS" \
    --lr "$LR" \
    --time_steps "$T" \
    --homeo_target weight \
    --eta_stdp 0.01 \
    --name "CARE_Plain${DEPTH}_Weight_Strong" \
    --output_dir "$RESULTS_DIR" \
    --block sew # ignored for plain arch

echo "V6 Deep Scale Suite finished executing! Check $RESULTS_DIR for results."
