#!/bin/bash
#SBATCH --job-name=ExplicitEval
#SBATCH --output=logs/ExplicitEval.out
#SBATCH --error=logs/ExplicitEval.err
#SBATCH --partition=a6000

mkdir -p logs

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Working dir: $(pwd)"

source ~/.bashrc
conda activate base

python 3_ExplicitBias_Evaluation.py \
    --models mistral gemma llama qwen phi \
    --exp both

echo "End time: $(date)"