#!/bin/bash
# Train ITHP + DeBERTa + Neuroplastic + Graph Fusion on MOSI

cd /home/gmn/danger/codes/ITHP
source /home/anaconda/etc/profile.d/conda.sh
conda activate ITHP

# 日志文件
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs"
mkdir -p $LOG_DIR
LOG_FILE="${LOG_DIR}/mosi_${TIMESTAMP}.log"

echo "Training on MOSI with Graph Fusion"
echo "Graph: 66 nodes (50 text + 8 visual + 8 acoustic)"
echo "Dataset: MOSI (1281 train samples)"
echo "Log: $LOG_FILE"
echo "=================================================="

nohup env CUDA_VISIBLE_DEVICES=1 python -u ITHP_SRigL/train_neuroplastic.py \
    --dataset mosi \
    --n_epochs 60 \
    --train_batch_size 32 \
    --learning_rate 3e-05 \
    --warmup_proportion 0.1 \
    --IB_coef 0.02 \
    --drop_prob 0.35 \
    --p_lambda 0.4 \
    --warmup_steps 300 \
    --max_prune_ratio 0.05 \
    --growth_ratio 0.04 \
    --use_graph_fusion \
    --graph_n_heads 4 \
    --graph_n_layers 2 \
    --graph_hidden_dim 256 \
    --n_visual_segments 8 \
    --n_acoustic_segments 8 \
    --sw_k 6 \
    --sw_p 0.15 \
    --sf_m 3 \
    --topology_alpha 0.5 \
    --cross_modal_connectivity 0.3 \
    --topology_coef 0.01 \
    --output_dir ./mosi_checkpoints \
    > "$LOG_FILE" 2>&1 &

echo "Training started in background, PID: $!"
echo "Monitor: tail -f $LOG_FILE"
