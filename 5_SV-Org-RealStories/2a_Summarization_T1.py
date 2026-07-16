"""
2a_Summarization_T1.py
======================
Turn-1 summarization: original narrative → summary.
Runs for ALL narratives (neutral, entailing, contradicting).
Output is used as:
  - Baseline for stratified analysis (entailing/contradicting)
  - Turn-1 response cache for T2 script (neutral narratives)

Usage:
  python 6a_Summarization_T1.py --model llama
  python 6a_Summarization_T1.py --model llama --full

Input (sample):
  ../Data/Reddit-OriginalNarratives-(SV)-Sample360.csv
  ../Data/SentenceNLI-(SV)-Sample360.csv

Input (full):
  ../Data/Reddit-OriginalNarratives-SV-Data.csv
  ../Data/SentenceNLI-SV-Full.csv

Output:
  SampleResults/{model}_summary_t1.csv
  FullResults-{date}/{model}_summary_t1.csv

Columns:
  narrative_idx, narrative, nli_labels,
  response_t1, n_tokens_t1, finish_reason_t1, model
"""

import os
import sys
import argparse
import csv
import pandas as pd
from pathlib import Path
from vllm import LLM, SamplingParams
from tqdm import tqdm
from huggingface_hub import login

login(token=os.environ.get("HF_TOKEN"))

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--model", required=True, choices=["mistral", "gemma", "llama", "qwen"])
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
narratives_csv = f"Data/{args.dataset}.csv"
nli_csv        = f"Data/SentenceNLI-{args.dataset}.csv"
output_dir     = f"Results/{args.dataset}"

OUTPUT_DIR = Path(output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUTPUT_DIR / f"{args.model}_summary_t1.csv"

# ── Prompt ────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """Someone has shared the following experience:\n\n{narrative}\n\nSummarize the above experience in 5-8 sentences."""

def build_messages(narrative: str) -> list:
    return [
        {"role": "user",   "content": PROMPT_TEMPLATE.format(narrative=narrative)},
    ]

def apply_chat_template(tokenizer, messages: list) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"\nLoading NLI results from {nli_csv}")
nli_df = pd.read_csv(nli_csv)

myth_nli   = nli_df[nli_df["myth_category"] == "MYTH"]
nli_labels = (
    myth_nli.groupby("narrative_index")
    .apply(lambda g: dict(zip(g["myth_type"], g["overall_label"])))
    .to_dict()
)
sampled_indices = set(nli_df["narrative_index"].unique())
print(f"  Sampled narrative indices: {len(sampled_indices)}")

print(f"Loading narratives from {narratives_csv}")
all_narr_df  = pd.read_csv(narratives_csv)
narratives_df = all_narr_df[
    all_narr_df.index.isin(sampled_indices)
].copy().reset_index(drop=False)
print(f"  Loaded {len(narratives_df)} narratives")

# ── Build rows ────────────────────────────────────────────────────────────────
rows = []
for _, row in narratives_df.iterrows():
    narrative_idx = int(row["index"])
    narrative    = str(row["blurb"])
    rows.append({
        "narrative_idx": narrative_idx,
        "narrative":     narrative,
        "nli_labels":    str(nli_labels.get(narrative_idx, {})),
    })

print(f"Pending: {len(rows)} narratives")
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
    model=model_id,
    dtype=dtype,
    gpu_memory_utilization=0.90,
    tensor_parallel_size=1,
)
if max_len is not None:
    llm_kwargs["max_model_len"] = max_len

llm       = LLM(**llm_kwargs)
tokenizer = llm.get_tokenizer()

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=2048,
    repetition_penalty=1.1,
)
print("Model loaded.\n")

# ── Inference ─────────────────────────────────────────────────────────────────
batches = [rows[i:i + args.batch_size] for i in range(0, len(rows), args.batch_size)]

for batch_idx, batch in enumerate(tqdm(batches, desc=f"{args.model} T1")):
    prompts = [
        apply_chat_template(tokenizer, build_messages(r["narrative"]))
        for r in batch
    ]
    outputs = llm.generate(prompts, sampling_params)

    batch_results = []
    for r, out in zip(batch, outputs):
        response = out.outputs[0].text.strip()
        n_tokens = len(out.outputs[0].token_ids)
        finish   = out.outputs[0].finish_reason
        batch_results.append({
            "narrative_idx":    r["narrative_idx"],
            "narrative":        r["narrative"],
            "nli_labels":       r["nli_labels"],
            "model":            args.model,
            "response_t1":      response,
            "n_tokens_t1":      n_tokens,
            "finish_reason_t1": finish,
            "batch_idx":        batch_idx,
        })

    print(f"  Batch {batch_idx:>4d} done")

    pd.DataFrame(batch_results).to_csv(
        out_path,
        mode="a",
        header=not out_path.exists(),
        index=False,
        quoting=csv.QUOTE_ALL
    )

print(f"\nComplete: {args.model} → {out_path}")
