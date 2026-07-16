#!/bin/bash
#SBATCH --job-name=RealStoriesEval
#SBATCH --output=logs/RealStoriesEval.out
#SBATCH --error=logs/RealStoriesEval.err

# Activate Conda environment
source ~/.bashrc

# Run Python script

python 4a_T2_Posthoc.py --dataset JustDetention
python 4a_T2_Posthoc.py --dataset SurvivorStories
