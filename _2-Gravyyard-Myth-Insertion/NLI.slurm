#!/bin/bash
#SBATCH --job-name=NLI
#SBATCH --output=log/NLI.out
#SBATCH --error=log/NLI.err
#SBATCH --partition=a4000

# Run the script

conda init
source ~/.bashrc
conda activate summary_env
python 2_NLI.py

