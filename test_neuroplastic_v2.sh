#!/bin/bash
# Test v2 neuroplastic model

cd /home/gmn/danger/codes/ITHP
source /home/anaconda/etc/profile.d/conda.sh
conda activate ITHP

CUDA_VISIBLE_DEVICES=1 python ITHP_SRigL/test_neuroplastic.py \
    --checkpoint neuroplastic_checkpoints_v2/best_model.pt \
    --dataset mosi
