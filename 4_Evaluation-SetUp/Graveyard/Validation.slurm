#!/bin/bash
#SBATCH --job-name=T1_normality
#SBATCH --output=logs/T1_normality.out
#SBATCH --error=logs/T1_normality.err
#SBATCH --partition=h100

# Activate Conda environment
source ~/.bashrc
conda activate summary_env

# Run Python script
python T1_normality.py --task AdviceGeneration
python T1_normality.py --task Summarization