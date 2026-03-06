#!/bin/bash
# ITHP + SRigL Training Script
# Usage: ./train_srigl.sh <GPU_ID> [OPTIONS]
# Example: ./train_srigl.sh 1 --dataset mosi --dense_allocation 0.1

set -e

# Default GPU
GPU_ID=${1:-0}
shift 2>/dev/null || true

# Get script directory (ITHP_SRigL/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Parent directory (ITHP/)
ITHP_DIR="$(dirname "$SCRIPT_DIR")"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate Jacob_SRigL 2>/dev/null || echo "Warning: Could not activate Jacob_SRigL environment"

# Change to ITHP directory (for dataset loading)
cd "$ITHP_DIR"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$SCRIPT_DIR/logs/train_${TIMESTAMP}.log"

echo "========================================"
echo "ITHP + SRigL Training"
echo "========================================"
echo "GPU: $GPU_ID"
echo "ITHP Directory: $ITHP_DIR"
echo "Log file: $LOG_FILE"
echo "Additional args: $@"
echo "========================================"

# Run training with nohup
CUDA_VISIBLE_DEVICES=$GPU_ID nohup python -u "$SCRIPT_DIR/train_srigl.py" \
    --output_dir "$SCRIPT_DIR/checkpoints" \
    "$@" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Training started with PID: $PID"
echo "$PID" > "$SCRIPT_DIR/logs/train_${TIMESTAMP}.pid"

# Show initial output
sleep 3
echo ""
echo "Initial output:"
echo "----------------------------------------"
head -50 "$LOG_FILE" 2>/dev/null || echo "Waiting for output..."
echo "----------------------------------------"
echo ""
echo "To monitor: tail -f $LOG_FILE"
echo "To stop: kill $PID"
