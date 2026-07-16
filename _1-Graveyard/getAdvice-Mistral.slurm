#!/bin/bash
#SBATCH --job-name=ad-mistral
#SBATCH --output=log/advice_mistral.out
#SBATCH --error=log/advice_mistral.err
#SBATCH --partition=a6000

echo "=== mistral Advice Generation ==="
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"

source ~/.bashrc
conda activate summary_env

python 5_AdviceGeneration.py \
    --model mistral

echo "Finished: $(date)"