#!/bin/bash
#SBATCH --job-name=RealStories
#SBATCH --output=logs/RealStories.out
#SBATCH --error=logs/RealStories.err
#SBATCH --partition=h100

# Activate Conda environment
source ~/.bashrc
conda activate summary_env

# Run Python script

python 2b_Summarization_T2.py --model mistral --dataset JustDetention
python 2b_Summarization_T2.py --model gemma --dataset JustDetention
python 2b_Summarization_T2.py --model llama --dataset JustDetention
python 2b_Summarization_T2.py --model qwen --dataset JustDetention

python 2b_Summarization_T2.py --model mistral --dataset SurvivorStories
python 2b_Summarization_T2.py --model gemma --dataset SurvivorStories
python 2b_Summarization_T2.py --model llama --dataset SurvivorStories
python 2b_Summarization_T2.py --model qwen --dataset SurvivorStories