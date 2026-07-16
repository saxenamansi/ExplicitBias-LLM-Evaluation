#!/bin/bash
#SBATCH --job-name=NSP
#SBATCH --output=log/NSP.out
#SBATCH --error=log/NSP.err
#SBATCH --partition=a4000

# Run the script

conda init
source ~/.bashrc
conda activate subspace_proj_env
python 4_ChooseBestCandidate.py

