#!/bin/bash
#SBATCH --job-name=ad-phi
#SBATCH --output=log/advice_phi.out
#SBATCH --error=log/advice_phi.err
#SBATCH --partition=a6000

echo "=== phi Advice Generation ==="
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"

source ~/.bashrc
conda activate summary_env

python 5_AdviceGeneration.py \
    --model phi

echo "Finished: $(date)"