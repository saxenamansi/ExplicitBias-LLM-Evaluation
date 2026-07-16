import os
import sys
import subprocess

MODELS = ["mistral", "gemma", "llama", "qwen", "phi"]
STAGES = [
    "iat_logprob.py",
]
N_GPUS = 1

def run_stage(script, skip_models=None):
    models_to_run = [m for m in MODELS if skip_models is None or m not in skip_models]
    failed = []

    for batch_start in range(0, len(models_to_run), N_GPUS):
        batch = models_to_run[batch_start : batch_start + N_GPUS]
        procs = {}
        for gpu_slot, model in enumerate(batch):
            cmd = [sys.executable, script, "--model", model]  # no --full = sample
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_slot)}
            print(f"  GPU {gpu_slot}: {script} --model {model}")
            procs[model] = subprocess.Popen(cmd, env=env)

        for model, proc in procs.items():
            proc.wait()
            if proc.returncode != 0:
                failed.append(model)
                print(f"  ERROR: {model} exited with code {proc.returncode}")

    return failed

for script in STAGES:
    print(f"\n=== {script} ===")
    failed = run_stage(script)
    if failed:
        print(f"  Failed: {failed}")

print("\nDone.")