#!/bin/bash
#SBATCH --job-name=ProjectAdvice
#SBATCH --output=log/3_ProjectAdvice.out
#SBATCH --error=log/3_ProjectAdvice.err

conda init
source /mnt/beegfs/msaxena4/miniconda/etc/profile.d/conda.sh
conda activate summary_env

echo "Job started: $(date)"
echo "Node: $(hostname)"

python 3_ProjectAdvice.py

echo "Job finished: $(date)"
