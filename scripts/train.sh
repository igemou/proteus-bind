#!/bin/bash
#SBATCH -p rsingh47-gcondo
#SBATCH --job-name=Train_ProteusRBP
#SBATCH --output=logs/train_proteus_%j.out
#SBATCH --error=logs/train_proteus_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

module load cuda/12.1
source /oscar/home/igemou/proteus-bind/.env/bin/activate

NUM_RBPS=1

python train.py \
    --train data/HNRNPC/train_split.pkl \
    --val data/HNRNPC/val_split.pkl \
    --pos_label data/HNRNPC/HNRNPC_positive_label.pkl \
    --neg_label data/HNRNPC/HNRNPC_negative_label.pkl \
    --save checkpoints/best_proteus_HNRNPC.pt \
    --epochs 50 \
    --batch 32 \
    --lr 1e-4 \
    --patience 7 \
    --num_rbps $NUM_RBPS

echo "=== Done! Check logs/train_proteus_${SLURM_JOB_ID}.out ==="
