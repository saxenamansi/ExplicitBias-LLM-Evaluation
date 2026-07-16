# import os
# import sys
# import subprocess

# MODELS = ["mistral", "gemma", "llama", "qwen", "phi"]

# STAGES = [
#     "1a_AdviceGeneration_T1.py",
#     "1b_AdviceGeneration_T2.py"
# ]

# def run_stage(script):
#     procs = {}
#     for i, model in enumerate(MODELS):
#         cmd = [sys.executable, script, "--model", model]
#         env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}
#         print(f"  GPU {i}: {script} --model {model}")
#         procs[model] = subprocess.Popen(cmd, env=env)

#     failed = []
#     for model, proc in procs.items():
#         proc.wait()
#         if proc.returncode != 0:
#             failed.append(model)
#             print(f"  ERROR: {model} exited with code {proc.returncode}")
#     return failed

# for script in STAGES:
#     print(f"\n=== {script} ===")
#     failed = run_stage(script)
#     if failed:
#         print(f"  Failed: {failed}")

import os
import sys
import subprocess

MODELS = ["mistral", "gemma", "llama", "qwen", "phi"]

STAGES = [
    "2_AdviceGeneration/1a_AdviceGeneration_T1.py",
    "2_AdviceGeneration/1b_AdviceGeneration_T2.py",
]

def run_stage(script, extra_args=[]):
    procs = {}
    for i, model in enumerate(MODELS):
        cmd = [sys.executable, script, "--model", model] + extra_args
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}
        print(f"  GPU {i}: {script} --model {model} {' '.join(extra_args)}")
        procs[model] = subprocess.Popen(cmd, env=env)

    failed = []
    for model, proc in procs.items():
        proc.wait()
        if proc.returncode != 0:
            failed.append(model)
            print(f"  ERROR: {model} exited with code {proc.returncode}")
    return failed

for script in STAGES:
    print(f"\n=== {script} ===")
    failed = run_stage(script, extra_args=["--max_rows", "2"])
    if failed:
        print(f"  Failed: {failed}")
        sys.exit(1)