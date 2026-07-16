#!/bin/bash
#SBATCH --job-name=ad-gemma
#SBATCH --output=log/advice_gemma.out
#SBATCH --error=log/advice_gemma.err
#SBATCH --partition=a6000

echo "=== gemma Advice Generation ==="
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"

source ~/.bashrc
conda activate summary_env

python 5_AdviceGeneration.py \
    --model gemma

echo "Finished: $(date)"