#!/bin/bash
#SBATCH --job-name=ad-${MODEL}
#SBATCH --output=log/advice_%x.out
#SBATCH --error=log/advice_%x.err
#SBATCH --partition=a100

echo "=== ${MODEL} Advice Generation ==="
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"

source ~/.bashrc
conda activate summary_env

python 5_AdviceGeneration.py --model ${MODEL}

echo "Finished: $(date)"