#!/usr/bin/env bash
# ============================================================
# Disconnect-Proof Experiment Launcher
# ============================================================
# This script launches experiments inside a tmux session so they
# survive SSH disconnects (e.g., VPN drops).
#
# Usage:
#   bash scripts/launch_experiments.sh          # Launch V3 suite
#   bash scripts/launch_experiments.sh --attach  # Launch & attach
#
# To re-attach later:
#   tmux attach -t care_experiments
#
# To check status:
#   tmux ls
#   tail -f v3_suite.log
# ============================================================

set -euo pipefail

# Fix for PyTorch Allocator Fragmentation (RTX 4090)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SESSION_NAME="care_experiments"
LOG_FILE="${PROJECT_DIR}/v3_suite.log"
SCRIPT_V3_SUITE="${PROJECT_DIR}/scripts/run_v3_suite.py"
SCRIPT_ABLATION="${PROJECT_DIR}/scripts/run_v3_rigorous_ablation.py"
ATTACH=false

# Parse args
for arg in "$@"; do
    case $arg in
        --attach) ATTACH=true ;;
    esac
done

# Check if session already exists
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "⚠️  tmux session '${SESSION_NAME}' already exists!"
    echo "   To attach:  tmux attach -t ${SESSION_NAME}"
    echo "   To kill:    tmux kill-session -t ${SESSION_NAME}"
    exit 1
fi

echo "============================================================"
echo "  CARE Experiment Launcher (disconnect-proof)"
echo "============================================================"
echo "  Project:  ${PROJECT_DIR}"
echo "  Script 1: ${SCRIPT_V3_SUITE}"
echo "  Script 2: ${SCRIPT_ABLATION}"
echo "  Log:      ${LOG_FILE}"
echo "  Session:  ${SESSION_NAME}"
echo "============================================================"

# Create a new detached tmux session and run the experiment
tmux new-session -d -s "${SESSION_NAME}" -c "${PROJECT_DIR}" \
    "echo '=== CARE V3 Experiment Suite ===' && \
     echo 'Started at: $(date)' && \
     echo 'Log: ${LOG_FILE}' && \
     echo '================================' && \
     # Skipping old suite to prioritize calibrated ablation
     # python ${SCRIPT_V3_SUITE} 2>&1 | tee ${LOG_FILE}; 
     echo '=== CARE V3 Rigorous Ablation Suite ===' 2>&1 | tee ${LOG_FILE}; \
     python ${SCRIPT_ABLATION} 2>&1 | tee -a ${LOG_FILE}; \
     echo ''; \
     echo '=== ALL EXPERIMENTS FINISHED ==='; \
     echo 'Finished at: $(date)'; \
     echo 'Press Enter to close this session...'; \
     read"

echo ""
echo "✅ Experiments launched in tmux session '${SESSION_NAME}'"
echo ""
echo "  📋 Useful commands:"
echo "     tmux attach -t ${SESSION_NAME}    # Watch live output"
echo "     tmux ls                            # List sessions"
echo "     tail -f ${LOG_FILE}           # Follow the log"
echo "     tmux kill-session -t ${SESSION_NAME}  # Kill if needed"
echo ""

if [ "$ATTACH" = true ]; then
    echo "Attaching to session..."
    tmux attach -t "${SESSION_NAME}"
fi
