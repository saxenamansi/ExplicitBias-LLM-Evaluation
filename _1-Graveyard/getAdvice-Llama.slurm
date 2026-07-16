#!/bin/bash
#SBATCH --job-name=ad-llama
#SBATCH --output=log/advice_llama.out
#SBATCH --error=log/advice_llama.err
#SBATCH --partition=a6000

echo "=== llama Advice Generation ==="
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"

source ~/.bashrc
conda activate summary_env

python 5_AdviceGeneration.py \
    --model llama

echo "Finished: $(date)"