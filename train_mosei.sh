#!/bin/bash
# Train ITHP + DeBERTa + Neuroplastic + Graph Fusion on MOSEI
# MOSEI: 16265 train, 1869 dev, 4643 test (much larger than MOSI)

cd /home/gmn/danger/codes/ITHP
source /home/anaconda/etc/profile.d/conda.sh
conda activate ITHP

echo "Training on MOSEI with Graph Fusion"
echo "Graph: 66 nodes (50 text + 8 visual + 8 acoustic)"
echo "Dataset: MOSEI (16K train samples)"
echo "=================================================="

CUDA_VISIBLE_DEVICES=1 python -u ITHP_SRigL/train_neuroplastic.py \
    --dataset mosei \
    --n_epochs 30 \
    --train_batch_size 32 \
    --learning_rate 3e-05 \
    --warmup_proportion 0.1 \
    --IB_coef 0.02 \
    --drop_prob 0.35 \
    --p_lambda 0.4 \
    --warmup_steps 800 \
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
    --output_dir ./mosei_checkpoints
