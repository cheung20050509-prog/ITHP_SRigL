#!/bin/bash
# Optuna hyperparameter optimization for ITHP + DeBERTa + Neuroplastic V3
# Key improvements over V2:
#   - Added weight_decay (L2 regularization) to reduce overfitting
#   - Added label_smoothing (noise on labels) for regularization
#   - Added early_stopping to prevent overfitting
#   - Increased dropout ranges (0.3-0.6)
#   - 100 epochs max with early stopping

cd /home/gmn/danger/codes/ITHP
source /home/anaconda/etc/profile.d/conda.sh
conda activate ITHP

# Install optuna if needed
pip install optuna -q 2>/dev/null

mkdir -p ITHP_SRigL/optuna_checkpoints_v3
mkdir -p ITHP_SRigL/logs

echo "Starting Optuna optimization V3 (with regularization)..."
echo "Results stored in: ITHP_SRigL/optuna_results_v3.db"
echo "NEW: weight_decay, label_smoothing, early_stopping"
echo ""

CUDA_VISIBLE_DEVICES=1 nohup python -u ITHP_SRigL/optuna_optimize.py \
    --n_trials 100 \
    --study_name ithp_neuroplastic_v3 \
    --db_path ITHP_SRigL/optuna_results_v3.db \
    > ITHP_SRigL/logs/optuna_v3.log 2>&1 &

echo "Started! PID: $!"
echo "Log: tail -f ITHP_SRigL/logs/optuna_v3.log"
echo ""
echo "To view study progress:"
echo "  optuna-dashboard sqlite:///ITHP_SRigL/optuna_results_v3.db"
