import os
import sys
import subprocess

MODELS = ["mistral", "gemma", "llama", "qwen", "phi"]
N_GPUS = 1

def run_stage(script, extra_args=None, skip_models=None):
    models_to_run = [m for m in MODELS if skip_models is None or m not in skip_models]
    extra_args    = extra_args or []
    failed        = []

    for batch_start in range(0, len(models_to_run), N_GPUS):
        batch  = models_to_run[batch_start : batch_start + N_GPUS]
        procs  = {}
        for gpu_slot, model in enumerate(batch):
            cmd = [sys.executable, script, "--model", model] + extra_args
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_slot)}
            print(f"  GPU {gpu_slot}: {script} --model {model} {' '.join(extra_args)}")
            procs[model] = subprocess.Popen(cmd, env=env)

        for model, proc in procs.items():
            proc.wait()
            if proc.returncode != 0:
                failed.append(model)
                print(f"  ERROR: {model} exited with code {proc.returncode}")

    return failed

# ── Stage 1: IAT test run ─────────────────────────────────────────────────────
print("\n=== IAT test run ===")
test_failed = run_stage("3a_TestRun-IAT.py")
if test_failed:
    print(f"  ⚠ Test failed for: {test_failed} — proceeding to main IAT regardless")

# ── Stage 2: IAT sample run ───────────────────────────────────────────────────
print("\n=== IAT run (sample set) ===")
iat_failed = run_stage("3_IATProb.py")
if iat_failed:
    print(f"  Failed: {iat_failed}")

print("\nDone.")