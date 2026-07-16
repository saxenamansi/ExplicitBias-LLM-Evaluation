#!/bin/bash
#SBATCH --job-name=GetNLI
#SBATCH --output=logs/GetNLI.out
#SBATCH --error=logs/GetNLI.err
#SBATCH --partition=a6000

# Activate Conda environment
source ~/.bashrc

# Run Python script
python NLI.py