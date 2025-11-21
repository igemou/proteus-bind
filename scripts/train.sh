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


python train.py \
    --mode single \
    --target_rbp 0 \
    --train data/HNRNPC/train_split.pkl,data/PCBP1/train_split.pkl,data/SRSF9/train_split.pkl,data/TIA1/train_split.pkl,data/TRA2A/train_split.pkl \
    --val   data/HNRNPC/val_split.pkl,data/PCBP1/val_split.pkl,data/SRSF9/val_split.pkl,data/TIA1/val_split.pkl,data/TRA2A/val_split.pkl \
    --pos_label data/HNRNPC/HNRNPC_positive_label.pkl,data/PCBP1/PCBP1_positive_label.pkl,data/SRSF9/SRSF9_positive_label.pkl,data/TIA1/TIA1_positive_label.pkl,data/TRA2A/TRA2A_positive_label.pkl \
    --neg_label data/HNRNPC/HNRNPC_negative_label.pkl,data/PCBP1/PCBP1_negative_label.pkl,data/SRSF9/SRSF9_negative_label.pkl,data/TIA1/TIA1_negative_label.pkl,data/TRA2A/TRA2A_negative_label.pkl \
    --save checkpoints/best_proteus_HNRNPC_bindingonly.pt \
    --epochs 20 \
    --batch 32 \
    --lr 1e-4 \
    --patience 5 \
    --lambda_bind 1.0 \
    --lambda_func 0

python train.py \
    --mode single \
    --target_rbp 0 \
    --train "data/HNRNPC/train_split.pkl,data/PCBP1/train_split.pkl,data/SRSF9/train_split.pkl,data/TIA1/train_split.pkl,data/TRA2A/train_split.pkl" \
    --val   "data/HNRNPC/val_split.pkl,data/PCBP1/val_split.pkl,data/SRSF9/val_split.pkl,data/TIA1/val_split.pkl,data/TRA2A/val_split.pkl" \
    --pos_label "data/HNRNPC/HNRNPC_positive_label.pkl,data/PCBP1/PCBP1_positive_label.pkl,data/SRSF9/SRSF9_positive_label.pkl,data/TIA1/TIA1_positive_label.pkl,data/TRA2A/TRA2A_positive_label.pkl" \
    --neg_label "data/HNRNPC/HNRNPC_negative_label.pkl,data/PCBP1/PCBP1_negative_label.pkl,data/SRSF9/SRSF9_negative_label.pkl,data/TIA1/TIA1_negative_label.pkl,data/TRA2A/TRA2A_negative_label.pkl" \
    --save checkpoints/best_proteus_HNRNPC_all.pt \
    --epochs 20 \
    --batch 32 \
    --lr 1e-4 \
    --lambda_bind 1.0 \
    --lambda_func 0.5 \
    --lambda_next 0.5

python train.py \
  --mode multi \
  --train data/HNRNPC/train_split.pkl,data/PCBP1/train_split.pkl,data/SRSF9/train_split.pkl,data/TIA1/train_split.pkl,data/TRA2A/train_split.pkl \
  --val   data/HNRNPC/val_split.pkl,data/PCBP1/val_split.pkl,data/SRSF9/val_split.pkl,data/TIA1/val_split.pkl,data/TRA2A/val_split.pkl \
  --pos_label data/HNRNPC/HNRNPC_positive_label.pkl,data/PCBP1/PCBP1_positive_label.pkl,data/SRSF9/SRSF9_positive_label.pkl,data/TIA1/TIA1_positive_label.pkl,data/TRA2A/TRA2A_positive_label.pkl \
  --neg_label data/HNRNPC/HNRNPC_negative_label.pkl,data/PCBP1/PCBP1_negative_label.pkl,data/SRSF9/SRSF9_negative_label.pkl,data/TIA1/TIA1_negative_label.pkl,data/TRA2A/TRA2A_negative_label.pkl \
  --save checkpoints/best_proteus_multi5_bindingonly.pt \
  --epochs 20 \
  --batch 32 \
  --lr 1e-4 \
  --lambda_bind 1.0 \
  --lambda_func 0

echo "=== Done! Check logs/train_proteus_${SLURM_JOB_ID}.out ==="
