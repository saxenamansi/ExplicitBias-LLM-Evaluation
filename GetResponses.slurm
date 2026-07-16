#!/bin/bash
#SBATCH --job-name=GetSummary1stPOVa100
#SBATCH --output=logs/GetSummary1stPOVa100.out
#SBATCH --error=logs/GetSummary1stPOVa100.err
#SBATCH --partition=a100


# Load Python environment
cd /mnt/beegfs/msaxena4/6_Explicit-Implicit-Bias
source /mnt/beegfs/msaxena4/miniconda/etc/profile.d/conda.sh


# Activate Conda environment
conda activate summary_env

# Run Python script
python 3_Summarization/1a_Summarization_T1.py --model gemma --full
python 3_Summarization/1a_Summarization_T1.py --model llama --full
python 3_Summarization/1a_Summarization_T1.py --model mistral --full
python 3_Summarization/1a_Summarization_T1.py --model qwen --full

python 3_Summarization/1b_Summarization_T2.py --model gemma --full
python 3_Summarization/1b_Summarization_T2.py --model llama --full
python 3_Summarization/1b_Summarization_T2.py --model mistral --full
python 3_Summarization/1b_Summarization_T2.py --model qwen --full

