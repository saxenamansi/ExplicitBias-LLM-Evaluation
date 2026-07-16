#!/bin/bash
#SBATCH --job-name=IAT_test
#SBATCH --partition=a6000
#SBATCH --output=logs/IAT_test/.out
#SBATCH --error=logs/IAT_test/.err

mkdir -p logs/IAT_test

source ~/.bashrc
conda activate summary_env

for model in mistral gemma llama qwen phi; do
    echo "Running IAT test: $model"
    python 4a_TestRun-IAT.py --model $model \
        > logs/IAT_test/${model}.out \
        2> logs/IAT_test/${model}.err
    if [ $? -ne 0 ]; then
        echo "  ERROR: $model failed — check logs/IAT_test/${model}.err"
    else
        echo "  Done: $model"
    fi
done