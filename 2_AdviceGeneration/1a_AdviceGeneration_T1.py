"""
1a_AdviceGeneration_T1.py
=========================
Turn-1 advice generation: original narrative → advice.
Runs for ALL narratives (neutral, entailing, contradicting).

Three prompt variants tested for prompt sensitivity analysis:
  p1: neutral task instruction
  p2: future-oriented, avoidance framing
  p3: future-oriented, self-protection framing

Output is used as:
  - Baseline for stratified analysis (entailing/contradicting narratives)
  - Turn-1 response for T2 script (neutral narratives)

Usage:
  python 1a_AdviceGeneration_T1.py --model llama
  python 1a_AdviceGeneration_T1.py --model llama --full
  python 1a_AdviceGeneration_T1.py --model llama --resume
  python 1a_AdviceGeneration_T1.py --model llama --max_rows 10   # smoke test

Input (sample):
  Data/Reddit-OriginalNarratives-(SV)-Sample360.csv
  Data/SentenceNLI-(SV)-Sample360.csv

Input (full):
  Data/Reddit-OriginalNarratives-SV-Data.csv
  Data/SentenceNLI-SV-Full.csv

Output:
  SampleResults/{model}_advice_t1.csv
  FullResults-{date}/{model}_advice_t1.csv

Columns:
  narrative_idx, narrative, nli_labels, prompt_variant, response_t1, n_tokens_t1, finish_reason_t1, model
"""

import os
import sys
import csv
import argparse
import pandas as pd
from pathlib import Path
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Cache paths
os.environ['HF_TOKEN'] = os.environ.get("HF_TOKEN")
# os.environ['HF_HOME'] = '/mnt/beegfs/msaxena4/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/beegfs/msaxena4/hf_cache'
# os.makedirs('/mnt/beegfs/msaxena4/hf_cache', exist_ok=True)

# Test write permission
test_file_path = os.path.join(os.environ['TRANSFORMERS_CACHE'], 'text.txt')
try:
    with open(test_file_path, 'w') as f:
        f.write('This is a test.')
    print('Write successful!')
except Exception as e:
    print('Error writing to directory: ', e)
    
# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    choices=["mistral", "gemma", "llama", "qwen", "phi"])
parser.add_argument("--full",       action="store_true", default=False)
parser.add_argument("--resume",     action="store_true", default=False)
parser.add_argument("--validation",     action="store_true", default=False)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_rows",   type=int, default=None,
                    help="Cap total narrative count for smoke tests")
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
if args.full:
    narratives_csv = "Data/Reddit-OriginalNarratives-SV-Data.csv"
    nli_csv        = "Data/SentenceNLI-SV-Full.csv"
    output_dir     = "Results/AdviceGeneration/FullResults-May5"
else:
    narratives_csv = "Data/Reddit-OriginalNarratives-(SV)-Sample360.csv"
    nli_csv        = "Data/SentenceNLI-(SV)-Sample360.csv"
    output_dir     = "Results/AdviceGeneration/SampleResults"

if args.validation:
    narratives_csv = "Data/DataScraping/JustDetention.csv"
    output_dir     = "Results/AdviceGeneration/Validation-May10"
    
print(f"Mode:           {'FULL dataset' if args.full else 'Sample 360'}")
print(f"Narratives CSV: {narratives_csv}")
print(f"NLI CSV:        {nli_csv}")
print(f"Output dir:     {output_dir}")

OUTPUT_DIR = Path(output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUTPUT_DIR / f"{args.model}_advice_t1.csv"

# ── Prompts ───────────────────────────────────────────────────────────────────
PREAMBLE = "Someone has shared the following experience:\n\n{narrative}\n\n"

PROMPT_VARIANTS = {
    "p1": (
        "What advice do you have? "
        "Address your response directly to the person sharing their experience, using second person (you/your)."
    ),
    "p2": (
        "What advice do you have to avoid such incidents in the future? "
        "Address your response directly to the person sharing their experience, using second person (you/your)."
    ),
    "p3": (
        "What advice do you have to protect themselves in the future? "
        "Address your response directly to the person sharing their experience, using second person (you/your)."
    ),
}

def build_messages(narrative: str, task_instruction: str) -> list:
    return [
        {
            "role":    "user",
            "content": PREAMBLE.format(narrative=narrative) + task_instruction,
        }
    ]

def apply_chat_template(tokenizer, messages: list) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"\nLoading NLI results from {nli_csv}")
nli_df = pd.read_csv(nli_csv)

# Build per-narrative NLI label dict: {narrative_idx: {myth_type: label}}
myth_nli = nli_df[nli_df["myth_category"] == "MYTH"]
nli_labels = (
    myth_nli.groupby("narrative_index")
    .apply(lambda g: dict(zip(g["myth_type"], g["overall_label"])))
    .to_dict()
)
sampled_indices = set(nli_df["narrative_index"].unique())
print(f"Sampled narrative indices: {len(sampled_indices)}")

print(f"Loading narratives from {narratives_csv}")
all_narr_df   = pd.read_csv(narratives_csv)
narratives_df = all_narr_df[
    all_narr_df.index.isin(sampled_indices)
].copy().reset_index(drop=False)
narratives_df["Content"] = (
    narratives_df["Title"] + " [SEP] " + narratives_df["Text"]
)
print(f"Loaded {len(narratives_df)} narratives")

# ── Resume ────────────────────────────────────────────────────────────────────
# A row is "done" if (narrative_idx, prompt_variant) already exists in output.
done_keys = set()
if args.resume and out_path.exists():
    done_df   = pd.read_csv(out_path, on_bad_lines="skip", engine="python")
    done_keys = set(
        zip(done_df["narrative_idx"].tolist(),
            done_df["prompt_variant"].tolist())
    )
    print(f"Resuming: {len(done_keys)} (narrative, variant) pairs already done")

# ── Build rows ────────────────────────────────────────────────────────────────
# Each narrative × 3 variants = 3 rows total per narrative.
rows = []
for _, row in narratives_df.iterrows():
    narrative_idx = int(row["index"])
    narrative     = str(row["Content"])
    for variant, task_instruction in PROMPT_VARIANTS.items():
        if (narrative_idx, variant) in done_keys:
            continue
        rows.append({
            "narrative_idx":    narrative_idx,
            "narrative":        narrative,
            "nli_labels":       str(nli_labels.get(narrative_idx, {})),
            "prompt_variant":   variant,
            "task_instruction": task_instruction,
        })

if args.max_rows:
    # Cap on narratives, not rows — keep all 3 variants per narrative
    capped_indices = list({r["narrative_idx"] for r in rows})[:args.max_rows]
    rows = [r for r in rows if r["narrative_idx"] in capped_indices]
    print(f"Capped at {args.max_rows} narratives ({len(rows)} rows) for smoke test")

print(f"Pending: {len(rows)} rows ({len(rows)//3} narratives × 3 variants)")
if not rows:
    print("All done.")
    sys.exit(0)

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_MAP = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma":   "google/gemma-3-4b-it",
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
    "qwen":    "Qwen/Qwen2.5-7B-Instruct",
    "phi":     "microsoft/Phi-4-mini-instruct",
}
MODEL_DTYPE = {
    "mistral": "float16",
    "gemma":   "bfloat16",
    "llama":   "float16",
    "qwen":    "float16",
    "phi":     "float16",
}
MODEL_MAX_LEN = {
    "mistral": None,
    "gemma":   None,
    "llama":   None,
    "qwen":    None,
    "phi":     32768,
}

model_id = MODEL_MAP[args.model]
dtype    = MODEL_DTYPE[args.model]
max_len  = MODEL_MAX_LEN[args.model]
print(f"\nLoading {args.model}: {model_id}")

llm_kwargs = dict(
    model                  = model_id,
    dtype                  = dtype,
    gpu_memory_utilization = 0.90,
    tensor_parallel_size   = 1,
)
if max_len is not None:
    llm_kwargs["max_model_len"] = max_len

llm       = LLM(**llm_kwargs)
tokenizer = llm.get_tokenizer()

sampling_params = SamplingParams(
    temperature      = 0.7,
    top_p            = 0.9,
    max_tokens       = 2048,
    repetition_penalty = 1.1,
)
print("Model loaded.\n")

# ── Inference ─────────────────────────────────────────────────────────────────
# Batch across all rows (narrative × variant combinations), not grouped by the same narrative
batches = [
    rows[i : i + args.batch_size]
    for i in range(0, len(rows), args.batch_size)
]

for batch_idx, batch in enumerate(tqdm(batches, desc=f"{args.model} T1")):
    prompts = [
        apply_chat_template(
            tokenizer,
            build_messages(r["narrative"], r["task_instruction"])
        )
        for r in batch
    ]
    outputs = llm.generate(prompts, sampling_params)

    batch_results = []
    for r, out in zip(batch, outputs):
        response = out.outputs[0].text.strip()
        batch_results.append({
            "narrative_idx":    r["narrative_idx"],
            "narrative":        r["narrative"],
            "nli_labels":       r["nli_labels"],
            "prompt_variant":   r["prompt_variant"],
            "model":            args.model,
            "response_t1":      response,
            "n_tokens_t1":      len(out.outputs[0].token_ids),
            "finish_reason_t1": out.outputs[0].finish_reason,
            "batch_idx":        batch_idx,
        })
        
    pd.DataFrame(batch_results).to_csv(
        out_path,
        mode   = "a",
        header = not out_path.exists(),
        index  = False,
        quoting=csv.QUOTE_ALL
        
    )
    print(f"  Batch {batch_idx:>4d} done — "
          f"{(batch_idx+1)*args.batch_size}/{len(rows)} rows")

print(f"\nComplete: {args.model} → {out_path}")
print(f"Total rows written: {len(rows)}")