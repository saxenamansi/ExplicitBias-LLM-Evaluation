#!/bin/bash
#SBATCH --job-name=EvalNLI
#SBATCH --output=log/EvalNLI.out
#SBATCH --error=log/EvalNLI.err
#SBATCH --partition=a6000

source ~/.bashrc
conda activate summary_env

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python 5_EvaluateNLI.py

echo "Job finished: $(date)"
