#!/bin/bash
#SBATCH -p rsingh47-gcondo
#SBATCH --job-name=Eval_ProteusRBP
#SBATCH --output=logs/eval_proteus_%j.out
#SBATCH --error=logs/eval_proteus_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

echo "=== Starting Eval ==="

module load cuda/12.1
source /oscar/home/igemou/proteus-bind/.env/bin/activate

python eval.py \
    --model checkpoints/best_proteus_HNRNPC.pt \
    --test data/HNRNPC/test_split.pkl \
    --pos_label data/HNRNPC/HNRNPC_positive_label.pkl \
    --neg_label data/HNRNPC/HNRNPC_negative_label.pkl 

python eval.py \
  --model checkpoints/best_proteus_multi5.pt \
  --test data/HNRNPC/test_split.pkl,data/PCBP1/test_split.pkl,data/SRSF9/test_split.pkl,data/TIA1/test_split.pkl,data/TRA2A/test_split.pkl \
  --pos_label data/HNRNPC/HNRNPC_positive_label.pkl,data/PCBP1/PCBP1_positive_label.pkl,data/SRSF9/SRSF9_positive_label.pkl,data/TIA1/TIA1_positive_label.pkl,data/TRA2A/TRA2A_positive_label.pkl \
  --neg_label data/HNRNPC/HNRNPC_negative_label.pkl,data/PCBP1/PCBP1_negative_label.pkl,data/SRSF9/SRSF9_negative_label.pkl,data/TIA1/TIA1_negative_label.pkl,data/TRA2A/TRA2A_negative_label.pkl

echo "=== Done! Check logs/proteus_${SLURM_JOB_ID}.out ==="
