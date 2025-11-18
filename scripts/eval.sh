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

NUM_RBPS=1

python eval.py \
    --model checkpoints/best_proteus_HNRNPC.pt \
    --test data/HNRNPC/test_split.pkl \
    --pos_label data/HNRNPC/HNRNPC_positive_label.pkl \
    --neg_label data/HNRNPC/HNRNPC_negative_label.pkl \
    --num_rbps $NUM_RBPS

echo "=== Done! Check logs/proteus_${SLURM_JOB_ID}.out ==="
