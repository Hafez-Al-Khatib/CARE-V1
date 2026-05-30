#!/bin/bash
# ==========================================================
#  NeurIPS V6 Ablation Suite - CARE SNN
#  Cross-Architecture + Cross-Dataset Experiments
#
#  NEW vs V5: Tests CARE on VGG-SNN, Plain ConvNet, and
#  ResNet. Also adds CIFAR-100 and depth-scaling experiments.
#
#  Hardware: Single RTX 4090 (24GB VRAM)
#  Estimated runtime: ~4 days sequential
# ==========================================================
set -e

PYTHON="/opt/mariett_team/MariettEnv/bin/python"
RUNNER="scripts/run_flexible_experiment.py"
OUT_DIR="results/v6_neurips_cross_arch"
EPOCHS=150
T=8
BS=16
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
        --time_steps $T --batch_size $BS --seed $SEED \
        --dataset cifar10 --init normal "$@" \
        || echo "[FAILED] $NAME"
    echo " [$(date '+%Y-%m-%d %H:%M:%S')] Finished: $NAME"
    echo ""
}

# ==============================================================
#  SECTION A: Cross-Architecture Comparison (Table 1)
#  The core NeurIPS argument: CARE is universal
# ==============================================================
echo "============================================================"
echo " SECTION A: CROSS-ARCHITECTURE COMPARISON"
echo "============================================================"

# A.1 VGG-11 (No skip connections — strong CARE case)
run_exp "vgg11_control"  --arch vgg --depth 11 --no_plasticity
run_exp "vgg11_care"     --arch vgg --depth 11 --homeo_target gamma

# A.2 Plain ConvNet-8 (No skip, No BN — CARE's strongest showcase)
run_exp "plain8_control" --arch plain --depth 8 --no_plasticity
run_exp "plain8_care"    --arch plain --depth 8 --homeo_target weight

# A.3 SEW-ResNet-18 Baseline (reference from V5 suite)
# Already have A0_control running; just need CARE version for comparison
run_exp "sew18_care"     --arch resnet --depth 18 --block sew --homeo_target gamma --eta_stdp 0.0001

# ==============================================================
#  SECTION B: Dataset Generalization (Table 2)
#  Prove CARE works on harder tasks
# ==============================================================
echo "============================================================"
echo " SECTION B: DATASET GENERALIZATION (CIFAR-100)"
echo "============================================================"

run_exp "c100_sew18_control" --arch resnet --depth 18 --block sew --dataset cifar100 --no_plasticity
run_exp "c100_sew18_care"    --arch resnet --depth 18 --block sew --dataset cifar100 --homeo_target gamma --eta_stdp 0.0001
run_exp "c100_vgg11_control" --arch vgg --depth 11 --dataset cifar100 --no_plasticity
run_exp "c100_vgg11_care"    --arch vgg --depth 11 --dataset cifar100 --homeo_target gamma

# ==============================================================
#  SECTION C: Depth Scaling Study (Figure 3)
#  Does CARE matter more as depth increases?
# ==============================================================
echo "============================================================"
echo " SECTION C: DEPTH SCALING"
echo "============================================================"

run_exp "sew6_control"   --arch resnet --depth 6 --block sew --no_plasticity
run_exp "sew6_care"      --arch resnet --depth 6 --block sew --homeo_target gamma --eta_stdp 0.0001
run_exp "sew34_control"  --arch resnet --depth 34 --block sew --no_plasticity
run_exp "sew34_care"     --arch resnet --depth 34 --block sew --homeo_target gamma --eta_stdp 0.0001

# ==============================================================
#  SECTION D: Sabotage/Lazarus across architectures (Table 3)
#  Prove CARE rescues different architectures from death
# ==============================================================
echo "============================================================"
echo " SECTION D: CROSS-ARCH SABOTAGE (LAZARUS TEST)"
echo "============================================================"

run_exp "vgg11_sab_control" --arch vgg --depth 11 --init sabotage --no_plasticity
run_exp "vgg11_sab_care"    --arch vgg --depth 11 --init sabotage --homeo_target gamma
run_exp "plain8_sab_control" --arch plain --depth 8 --init sabotage --no_plasticity
run_exp "plain8_sab_care"   --arch plain --depth 8 --init sabotage --homeo_target weight

echo "=========================================================="
echo " NeurIPS V6 Cross-Architecture Suite Complete!"
echo " Total Experiments: 20"
echo " Results in: $OUT_DIR"
echo "=========================================================="
