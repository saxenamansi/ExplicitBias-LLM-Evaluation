#!/bin/bash
#SBATCH --job-name=GetResponseEmbeddings
#SBATCH --output=logs/GetResponseEmbeddings.out
#SBATCH --error=logs/GetResponseEmbeddings.err
#SBATCH --partition=a100

# Load Python environment
cd /mnt/beegfs/msaxena4
source /mnt/beegfs/msaxena4/miniconda/etc/profile.d/conda.sh
export PYTHONPATH=/mnt/beegfs/msaxena4:$PYTHONPATH

# Activate Conda environment
conda activate summary_env

# Run Python script

python 6_Explicit-Implicit-Bias/3_Summarization/2_GetResponseEmbeddings.py --source reddit

