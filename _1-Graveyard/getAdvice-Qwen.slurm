#!/bin/bash
#SBATCH --job-name=ad-qwen
#SBATCH --output=log/advice_qwen.out
#SBATCH --error=log/advice_qwen.err
#SBATCH --partition=a6000

echo "=== qwen Advice Generation ==="
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"

source ~/.bashrc
conda activate summary_env

python 5_AdviceGeneration.py \
    --model qwen

echo "Finished: $(date)"