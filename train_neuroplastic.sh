#!/bin/bash
# Neuroplastic training script

cd /home/gmn/danger/codes/ITHP
source /home/anaconda/etc/profile.d/conda.sh
conda activate ITHP

mkdir -p ITHP_SRigL/logs

CUDA_VISIBLE_DEVICES=1 nohup python ITHP_SRigL/train_neuroplastic.py \
    --dataset mosi \
    --n_epochs 30 \
    --train_batch_size 8 \
    --warmup_steps 500 \
    --prune_interval 300 \
    --growth_interval 300 \
    > ITHP_SRigL/logs/train_neuroplastic_mosi.log 2>&1 &

echo "Started! PID: $!"
echo "Log: ITHP_SRigL/logs/train_neuroplastic_mosi.log"
