#!/bin/bash
# Neuroplastic training v2 - 自适应拓扑更新
# 核心目标：通过神经可塑性优化来提升性能，而非稀疏化
# 
# 改进点：
# 1. 基于稳定性触发更新（等权重收敛后再改拓扑）
# 2. Net-Zero 重新布线（剪多少长多少）
# 3. 自适应调整更新幅度
# 4. 更长的训练周期

cd /home/gmn/danger/codes/ITHP
source /home/anaconda/etc/profile.d/conda.sh
conda activate ITHP

mkdir -p ITHP_SRigL/logs
mkdir -p neuroplastic_checkpoints_v2

CUDA_VISIBLE_DEVICES=1 nohup python -u ITHP_SRigL/train_neuroplastic.py \
    --dataset mosi \
    --n_epochs 60 \
    --train_batch_size 8 \
    --warmup_steps 800 \
    --max_prune_ratio 0.05 \
    --growth_ratio 0.05 \
    --checkpoint_dir neuroplastic_checkpoints_v2 \
    > ITHP_SRigL/logs/train_neuroplastic_v2_mosi.log 2>&1 &

echo "Started v2 training! PID: $!"
echo ""
echo "Key changes from v1:"
echo "  - epochs: 30 -> 60 (更长训练)"
echo "  - warmup: 500 -> 800 (让网络先稳定)"
echo "  - 基于稳定性触发拓扑更新（不再固定间隔）"
echo "  - Net-Zero: prune K + grow K（保持连接数）"
echo "  - 自适应调整更新幅度"
echo ""
echo "Log: tail -f ITHP_SRigL/logs/train_neuroplastic_v2_mosi.log"
