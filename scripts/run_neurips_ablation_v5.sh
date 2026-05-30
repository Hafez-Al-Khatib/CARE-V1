#!/bin/bash
# ==========================================================
#  NeurIPS V5 Ablation Suite - CARE SNN
#  5-Axis Ablation Matrix for Homeostatic Plasticity
#  
#  Axes:
#    1. Homeostatic Target (gamma / weight / both / none)
#    2. SNR Gating (off / threshold sweep)
#    3. Plasticity LR (eta_stdp sweep)
#    4. Block Architecture (SEW vs MS)
#    5. Initialization (normal vs sabotage)
#
#  Dataset: CIFAR-10 (primary), 150 epochs
#  Hardware: Single RTX 4090
# ==========================================================
set -e

PYTHON="/opt/mariett_team/MariettEnv/bin/python"
RUNNER="scripts/run_flexible_experiment.py"
OUT_DIR="results/v5_neurips_ablation"
EPOCHS=150
T=8
BS=16
DEPTH=18
SEED=42

mkdir -p $OUT_DIR

run_exp() {
    local NAME=$1
    shift
    echo "=========================================================="
    echo " [$(date '+%Y-%m-%d %H:%M:%S')] Starting: $NAME"
    echo "  Args: $@"
    echo "=========================================================="
    $PYTHON $RUNNER --name "$NAME" --output_dir "$OUT_DIR" --epochs $EPOCHS \
        --time_steps $T --batch_size $BS --depth $DEPTH --seed $SEED \
        --dataset cifar10 --init normal --block sew "$@" \
        || echo "[FAILED] $NAME"
    echo " [$(date '+%Y-%m-%d %H:%M:%S')] Finished: $NAME"
    echo ""
}

# ==============================================================
#  AXIS 0: Baseline Control (No Plasticity)
# ==============================================================
echo "============================================================"
echo " AXIS 0: BASELINE CONTROL"
echo "============================================================"
run_exp "A0_control" --no_plasticity

# ==============================================================
#  AXIS 1: Homeostatic Target (gamma vs weight vs both)
#  All use default SNR gating (threshold=2.0, steepness=5.0)
# ==============================================================
echo "============================================================"
echo " AXIS 1: HOMEOSTATIC TARGET"
echo "============================================================"
run_exp "A1_gamma"  --homeo_target gamma
run_exp "A1_weight" --homeo_target weight
run_exp "A1_both"   --homeo_target both

# ==============================================================
#  AXIS 2: SNR Gating Sweep (using gamma target)
# ==============================================================
echo "============================================================"
echo " AXIS 2: SNR GATING SWEEP"
echo "============================================================"
run_exp "A2_snr_off"      --snr_off
run_exp "A2_snr_low"      --snr_threshold 1.0
# A2_snr_default is same as A1_gamma, skip
run_exp "A2_snr_high"     --snr_threshold 4.0

# ==============================================================
#  AXIS 3: Plasticity LR Sweep (using gamma target, default SNR)
# ==============================================================
echo "============================================================"
echo " AXIS 3: PLASTICITY LR SWEEP"
echo "============================================================"
run_exp "A3_eta_1e-4" --eta_stdp 0.0001
run_exp "A3_eta_5e-4" --eta_stdp 0.0005
run_exp "A3_eta_1e-3" --eta_stdp 0.001
# A3_eta_5e-3 is the default (same as A1_gamma), skip

# ==============================================================
#  AXIS 4: Block Architecture (MS-ResNet)
# ==============================================================
echo "============================================================"
echo " AXIS 4: BLOCK ARCHITECTURE (MS-ResNet)"
echo "============================================================"
# Override block type to MS
$PYTHON $RUNNER --name "A4_ms_control" --output_dir "$OUT_DIR" --epochs $EPOCHS \
    --time_steps $T --batch_size $BS --depth $DEPTH --seed $SEED \
    --dataset cifar10 --init normal --block ms --no_plasticity \
    || echo "[FAILED] A4_ms_control"

$PYTHON $RUNNER --name "A4_ms_care" --output_dir "$OUT_DIR" --epochs $EPOCHS \
    --time_steps $T --batch_size $BS --depth $DEPTH --seed $SEED \
    --dataset cifar10 --init normal --block ms --homeo_target gamma \
    || echo "[FAILED] A4_ms_care"

# ==============================================================
#  AXIS 5: Initialization Regime (Sabotage = Lazarus Test)
# ==============================================================
echo "============================================================"
echo " AXIS 5: INITIALIZATION (SABOTAGE / LAZARUS)"
echo "============================================================"
$PYTHON $RUNNER --name "A5_sabotage_control" --output_dir "$OUT_DIR" --epochs $EPOCHS \
    --time_steps $T --batch_size $BS --depth $DEPTH --seed $SEED \
    --dataset cifar10 --init sabotage --block sew --no_plasticity \
    || echo "[FAILED] A5_sabotage_control"

$PYTHON $RUNNER --name "A5_sabotage_gamma" --output_dir "$OUT_DIR" --epochs $EPOCHS \
    --time_steps $T --batch_size $BS --depth $DEPTH --seed $SEED \
    --dataset cifar10 --init sabotage --block sew --homeo_target gamma \
    || echo "[FAILED] A5_sabotage_gamma"

$PYTHON $RUNNER --name "A5_sabotage_weight" --output_dir "$OUT_DIR" --epochs $EPOCHS \
    --time_steps $T --batch_size $BS --depth $DEPTH --seed $SEED \
    --dataset cifar10 --init sabotage --block sew --homeo_target weight \
    || echo "[FAILED] A5_sabotage_weight"

echo "=========================================================="
echo " NeurIPS V5 Ablation Suite Complete!"
echo " Total Experiments: 14"
echo " Results in: $OUT_DIR"
echo "=========================================================="
