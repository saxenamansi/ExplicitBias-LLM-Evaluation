#!/bin/bash
#SBATCH --job-name=EncodeResponses
#SBATCH --output=logs/EncodeResponses.out
#SBATCH --error=logs/EncodeResponses.err
#SBATCH --partition=a6000

# Activate Conda environment
source ~/.bashrc
conda activate summary_env

# Run Python script

python 3_EncodeResponses.py