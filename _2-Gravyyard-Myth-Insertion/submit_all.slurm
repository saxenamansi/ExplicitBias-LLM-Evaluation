#!/bin/bash
PARTITION=a100

for MODEL in llama mistral qwen gemma phi; do
    sbatch --partition=${PARTITION} \
           --job-name=ad-${MODEL} \
           --export=ALL,MODEL=${MODEL} \
           getAdvice.slurm
    echo "Submitted: ${MODEL}"
done