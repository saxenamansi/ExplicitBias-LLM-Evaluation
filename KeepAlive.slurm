#!/bin/bash
#SBATCH --job-name=Filler-GetFullResponses
#SBATCH --output=logs/Filler-GetFullResponses.out
#SBATCH --error=logs/Filler-GetFullResponses.err
#SBATCH --partition=a100
#SBATCH --nodelist=c33

# Load Python environment
cd /mnt/beegfs/msaxena4/6_Explicit-Implicit-Bias

# Activate Conda environment
source ~/.bashrc
conda activate summary_env

# Run Python script
python KeepAlive.py