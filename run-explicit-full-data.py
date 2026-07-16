import os
import sys
import subprocess

MODELS = ["mistral", "gemma", "llama", "qwen"]
          # "phi"]
STAGES = [
    # "1a_AdviceGeneration_T1.py",
    "1b_AdviceGeneration_T2.py",
    "1a_Summarization_T1.py",
    "1b_Summarization_T2.py",
]
N_GPUS = 5

# Define which stages are dependent: if key fails, skip value
DEPENDENCIES = {
    "1a_AdviceGeneration_T1.py": "1b_AdviceGeneration_T2.py",
    "2a_Summarization_T1.py":    "2b_Summarization_T2.py",
}

def run_stage(script, skip_models=None):
    models_to_run = [m for m in MODELS if skip_models is None or m not in skip_models]
    failed = []

    for batch_start in range(0, len(models_to_run), N_GPUS):
        batch = models_to_run[batch_start : batch_start + N_GPUS]
        procs = {}
        for gpu_slot, model in enumerate(batch):
            cmd = [sys.executable, script, "--model", model, "--full"]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_slot)}
            print(f"  GPU {gpu_slot}: {script} --model {model}")
            procs[model] = subprocess.Popen(cmd, env=env)

        for model, proc in procs.items():
            proc.wait()
            if proc.returncode != 0:
                failed.append(model)
                print(f"  ERROR: {model} exited with code {proc.returncode}")

    return failed

# Track which models to skip per dependent stage
skip_for_next = {}  # maps dependent script -> set of models to skip

for script in STAGES:
    skip_models = skip_for_next.get(script, set())
    if skip_models:
        print(f"\n=== {script} === (skipping {skip_models} due to prior failure)")
    else:
        print(f"\n=== {script} ===")

    failed = run_stage(script, skip_models=skip_models)

    if failed and script in DEPENDENCIES:
        dependent = DEPENDENCIES[script]
        skip_for_next.setdefault(dependent, set()).update(failed)
        print(f"  Propagating failures {failed} → skip in {dependent}")