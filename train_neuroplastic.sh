#!/bin/bash
# Neuroplastic training script
# UCB1 RL + Activity-based pruning + Hebbian growth + RigL cleanup

cd /home/gmn/danger/codes/ITHP
source /home/anaconda/etc/profile.d/conda.sh
conda activate ITHP

mkdir -p ITHP_SRigL/logs
mkdir -p neuroplastic_checkpoints

CUDA_VISIBLE_DEVICES=1 nohup python -u ITHP_SRigL/train_neuroplastic.py \
    --dataset mosi \
    --n_epochs 60 \
    --train_batch_size 32 \
    --warmup_steps 800 \
    --max_prune_ratio 0.05 \
    --growth_ratio 0.05 \
    --checkpoint_dir neuroplastic_checkpoints \
    > ITHP_SRigL/logs/train_neuroplastic_mosi.log 2>&1 &

echo "Started! PID: $!"
echo "Log: tail -f ITHP_SRigL/logs/train_neuroplastic_mosi.log"
