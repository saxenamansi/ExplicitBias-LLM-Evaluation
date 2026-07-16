"""
2b_Summarization_T2.py
======================
Turn-2 summarization: additional myth context → revised summary.
Reads T1 output. Runs only on neutral narratives per myth/pair filtering.

Filtering logic:
  - Single myth turn-2: narrative neutral for THAT myth only
  - Pair turn-2: narrative neutral for BOTH myths in the pair

Myth combos (80 total per neutral narrative):
  - 4 single myths × 4 frames × 2 doses = 32
  - 6 myth pairs × 4 frames × 2 doses   = 48

All 80 turn-2 variants for a given narrative are batched together
to exploit vLLM prefix caching (shared turn-1 prefix).

Usage:
  python 6b_Summarization_T2.py --model llama
  python 6b_Summarization_T2.py --model llama --full

Input:
  SampleResults/{model}_summary_t1.csv  (or FullResults/)
  ../Data/SentenceNLI-(SV)-Sample360.csv (or Full)

Output:
  SampleResults/{model}_summary_t2.csv
  FullResults-{date}/{model}_summary_t2.csv

Columns:
  narrative_idx, myth_type, myth_pair, frame, dose, myth_sentence, is_pair,
  response_t2, n_tokens_t2, finish_reason_t2, model
"""

import os
import sys
import argparse
import pandas as pd
from itertools import combinations
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
t1_path  = OUTPUT_DIR / f"{args.model}_summary_t1.csv"
out_path = OUTPUT_DIR / f"{args.model}_summary_t2.csv"

if not t1_path.exists():
    print(f"ERROR: T1 output not found at {t1_path}. Run 2a first.")
    sys.exit(1)

# ── Myth statements ───────────────────────────────────────────────────────────
try:
    from MythSentences import MYTH_SENTENCES
except ImportError:
    raise ImportError(
        "MythSentences.py not found. "
        "Please ensure myth statement dictionaries are defined there."
    )

MYTH_TYPES = ["clothing", "victim_intoxication", "perpetrator_intoxication", "resistance"]
MYTH_PAIRS = [
    tuple(sorted(pair)) for pair in combinations(MYTH_TYPES, 2)
]  # sorted alphabetically for consistent ordering

# ── Prompt ────────────────────────────────────────────────────────────────────
PROMPT_T1_TEMPLATE = """Someone has shared the following experience:\n\n{narrative}\n\nSummarize the above experience in 5-8 sentences."""

PROMPT_T2_TEMPLATE = "Here is more context:\n\n{context}\n\nBased on this, please provide a complete revised summary of the experience."

def build_t1_messages(narrative: str) -> list:
    return [
        {"role": "user",   "content": PROMPT_T1_TEMPLATE.format(narrative=narrative)},
    ]

def build_t2_messages(narrative: str, t1_response: str, context: str) -> list:
    return [
        {"role": "user",      "content": PROMPT_T1_TEMPLATE.format(narrative=narrative)},
        {"role": "assistant", "content": t1_response},
        {"role": "user",      "content": PROMPT_T2_TEMPLATE.format(context=context)},
    ]

def apply_chat_template(tokenizer, messages: list) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

# ── Load NLI labels ───────────────────────────────────────────────────────────
print(f"Loading NLI results from {nli_csv}")
nli_df   = pd.read_csv(nli_csv)
myth_nli = nli_df[nli_df["myth_category"] == "MYTH"]

nli_labels = (
    myth_nli.groupby("narrative_index")
    .apply(lambda g: dict(zip(g["myth_type"], g["overall_label"])))
    .to_dict()
)

def is_neutral(narrative_idx: int, myth_type: str) -> bool:
    labels = nli_labels.get(narrative_idx, {})
    return labels.get(myth_type, "neutral") == "neutral"

def row_key(narrative_idx, myth_type, myth_pair, frame, dose):
    return (narrative_idx, myth_type or "", myth_pair or "", frame, str(dose))

# ── Load T1 results ───────────────────────────────────────────────────────────
print(f"\nLoading T1 results from {t1_path}")
t1_df = pd.read_csv(t1_path, on_bad_lines="skip", engine="python")
print(f"  {len(t1_df)} narratives in T1 output")

# ── Build per-narrative work units ────────────────────────────────────────────
narrative_batches = []
narratives = t1_df.to_dict("records")

for row in narratives:
    narrative_idx = int(row["narrative_idx"])
    narrative     = str(row["narrative"])
    t1_response   = str(row["response_t1"])

    variants = []

    # ── Single myth variants ──────────────────────────────────────────────────
    for (myth_type, frame, dose), myth_sentence in MYTH_SENTENCES.items():
        if not is_neutral(narrative_idx, myth_type):
            continue
        variants.append({
            "narrative_idx":  narrative_idx,
            "myth_type":      myth_type,
            "myth_pair":      None,
            "frame":          frame,
            "dose":           dose,
            "myth_sentence":  myth_sentence,
            "is_pair":        False,
            "context":        myth_sentence,
        })

    # ── Myth pair variants ────────────────────────────────────────────────────
    for myth_a, myth_b in MYTH_PAIRS:
        if not (is_neutral(narrative_idx, myth_a) and is_neutral(narrative_idx, myth_b)):
            continue
        for (myth_type, frame, dose), myth_sentence in MYTH_SENTENCES.items():
            if myth_type != myth_a:
                continue
            stmt_a     = myth_sentence
            stmt_b     = MYTH_SENTENCES[(myth_b, frame, dose)]
            combined   = f"{stmt_a} {stmt_b}"
            pair_label = f"{myth_a}+{myth_b}"
            variants.append({
                "narrative_idx":  narrative_idx,
                "myth_type":      None,
                "myth_pair":      pair_label,
                "frame":          frame,
                "dose":           dose,
                "myth_sentence":  combined,
                "is_pair":        True,
                "context":        combined,
            })

    # ── Single myth variants ──────────────────────────────────────────────────
    # for dose, frame_dict in MYTH_SENTENCES.items():
    #     for frame, myth_type_dict in frame_dict.items():
    #         for myth_type, myth_sentence in myth_type_dict.items():
    #             if not is_neutral(narrative_idx, myth_type):
    #                 continue
    #             key = row_key(narrative_idx, myth_type, None, frame, dose)
    #             variants.append({
    #                 "narrative_idx":  narrative_idx,
    #                 "myth_type":      myth_type,
    #                 "myth_pair":      None,
    #                 "frame":          frame,
    #                 "dose":           dose,
    #                 "myth_sentence": myth_sentence,
    #                 "is_pair":        False,
    #                 "context":        myth_sentence,
    #             })

    # ── Myth pair variants ────────────────────────────────────────────────────
    # for myth_a, myth_b in MYTH_PAIRS:
    #     if not (is_neutral(narrative_idx, myth_a) and is_neutral(narrative_idx, myth_b)):
    #         continue
    #     for dose, frame_dict in MYTH_SENTENCES.items():
    #         for frame, myth_type_dict in frame_dict.items():
    #             stmt_a     = myth_type_dict[myth_a]
    #             stmt_b     = myth_type_dict[myth_b]
    #             combined   = f"{stmt_a} {stmt_b}"
    #             pair_label = f"{myth_a}+{myth_b}"
    #             key = row_key(narrative_idx, None, pair_label, frame, dose)
    #             variants.append({
    #                 "narrative_idx":  narrative_idx,
    #                 "myth_type":      None,
    #                 "myth_pair":      pair_label,
    #                 "frame":          frame,
    #                 "dose":           dose,
    #                 "myth_sentence": combined,
    #                 "is_pair":        True,
    #                 "context":        combined,
    #             })

    if variants:
        narrative_batches.append({
            "narrative_idx": narrative_idx,
            "narrative":     narrative,
            "t1_response":   t1_response,
            "variants":      variants,
        })

total_variants = sum(len(b["variants"]) for b in narrative_batches)
print(f"\nNarratives with pending T2 variants: {len(narrative_batches)}")
print(f"Total pending T2 prompts:            {total_variants:,}")

if not narrative_batches:
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
    enable_prefix_caching=True,
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

# ── Inference — per-narrative batches for prefix caching ─────────────────────
for narr_idx, narr_batch in enumerate(tqdm(narrative_batches, desc=f"{args.model} T2")):
    narrative_idx = narr_batch["narrative_idx"]
    narrative     = narr_batch["narrative"]
    t1_response   = narr_batch["t1_response"]
    variants      = narr_batch["variants"]

    prompts = [
        apply_chat_template(
            tokenizer,
            build_t2_messages(narrative, t1_response, v["context"])
        )
        for v in variants
    ]

    outputs = llm.generate(prompts, sampling_params)

    batch_results = []
    for v, out in zip(variants, outputs):
        response  = out.outputs[0].text.strip()
        n_tokens  = len(out.outputs[0].token_ids)
        finish    = out.outputs[0].finish_reason
        
        batch_results.append({
            "narrative_idx":    v["narrative_idx"],
            "myth_type":        v["myth_type"],
            "myth_pair":        v["myth_pair"],
            "frame":            v["frame"],
            "dose":             v["dose"],
            "myth_sentence":   v["myth_sentence"],
            "is_pair":          v["is_pair"],
            "model":            args.model,
            "response_t2":      response,
            "n_tokens_t2":      n_tokens,
            "finish_reason_t2": finish,
        })
    
    print(f"  Narrative {narrative_idx} | {len(variants)} variants")

    pd.DataFrame(batch_results).to_csv(
        out_path,
        mode="a",
        header=not out_path.exists(),
        index=False,
        quoting=1,
    )

print(f"\nComplete: {args.model} → {out_path}")
