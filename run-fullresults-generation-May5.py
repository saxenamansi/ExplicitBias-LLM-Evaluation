import os
import sys
import subprocess

MODELS = ["mistral", "gemma", "llama", "qwen"]
STAGES = [
    # "2_AdviceGeneration/1a_AdviceGeneration_T1.py",
    # "2_AdviceGeneration/1b_AdviceGeneration_T2.py",
    "3_Summarization/1a_Summarization_T1.py",
    "3_Summarization/1b_Summarization_T2.py",
]

def run_stage(script, skip_models=None):
    skip_models = skip_models or []
    failed = []
    for model in MODELS:
        if model in skip_models:
            continue
        cmd = [sys.executable, script, "--model", model, "--full"]
        print(f"  Running: {script} --model {model}")
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            failed.append(model)
            print(f"  ERROR: {model} exited with code {proc.returncode}")
    return failed

failed = []
for script in STAGES:
    print(f"\n=== {script} ===")
    failed = run_stage(script, skip_models=failed)
    if failed:
        print(f"  Failed: {failed}")