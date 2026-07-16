#!/bin/bash
#SBATCH --job-name=Summary_1stPOV_Analysis
#SBATCH --output=logs/Summary_1stPOV_Analysis.out
#SBATCH --error=logs/Summary_1stPOV_Analysis.err

# Activate Conda environment
source ~/.bashrc
conda activate summary_env

# Run Python script

python 2c_T1_Dunn.py --task Summarization --analysis MYTHS
python 2c_T1_Dunn.py --task Summarization --analysis DEMO

python 3a_T2-NormalityCheck.py --task Summarization 
python 3b_T2-MythAlignment-Paired-Wilcoxon.py --task Summarization 