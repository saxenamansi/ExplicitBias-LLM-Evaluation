"""
Build T1/T2 embeddings for Advice Generation.

T1: all narratives — one row per (narrative_idx, model, prompt_variant)
T2: neutral-only narratives — one row per (narrative_idx, model, prompt_variant, myth_type/pair, frame, dose)
Each T2 row is matched to exactly one T1 row; halts if any T2 has no T1 match.

Outputs per model: {model}_t1.pkl, {model}_t2_pairs.pkl, {model}_t1_metadata.csv, {model}_t2_metadata.csv
Directory: /{SampleResults/FullResults}/1_Embeddings/{SBERT|W2V|GLOVE}/)
  
"""
import sys
import argparse
import pandas as pd
import importlib.util
from pathlib import Path
import importlib.util

def load_module(name, rel_path):
    path = Path(__file__).parent.parent / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

Config    = load_module("Config",    "4_Evaluation-SetUp/Config.py")
Embeddings = load_module("Embeddings", "4_Evaluation-SetUp/Embeddings.py")

NLI_CSV_FULL    = Config.NLI_CSV_FULL
encode_sbert      = Embeddings.encode_sbert
get_w2v           = Embeddings.get_w2v
get_glove         = Embeddings.get_glove
mean_word_embedding = Embeddings.mean_word_embedding

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None)  # None = run all 5
parser.add_argument("--t1",    default=None)
parser.add_argument("--t2",    default=None)
parser.add_argument("--nli",   default=str(NLI_CSV_FULL))
parser.add_argument("--source", choices=["reddit", "JustDetention", "SurvivorStories"], default="reddit")
args = parser.parse_args()

_TASK   = Path(__file__).parent                        
_ROOT = Path(__file__).parent.parent

if args.source == "reddit":
    BASE    = _TASK / "../Results/AdviceGeneration/FullResults-May5"
    OUT_DIR = _TASK / "../Results/AdviceGeneration/1_Embeddings"
    NLI     = args.nli
else:
    BASE    = _ROOT / f"SV-Org-RealStories/Results/{args.source}"
    OUT_DIR = _ROOT / f"SV-Org-RealStories/Results/{args.source}/1_Embeddings"
    NLI     = str(_ROOT / f"SV-Org-RealStories/Data/SentenceNLI-{args.source}.csv")

MODELS = ["gemma", "llama", "mistral", "qwen"] if args.model is None else [args.model]
EMBS    = ["SBERT", "W2V", "GLOVE"]
for emb in EMBS:
    (OUT_DIR / emb).mkdir(parents=True, exist_ok=True)

# ── Embedding Models ──────────────────────────────────────────────────────────
w2v, glove = get_w2v(), get_glove()

def embed_all(texts):
    return {
        "SBERT": list(encode_sbert(texts)),
        "W2V":   [mean_word_embedding(t, w2v)   for t in texts],
        "GLOVE": [mean_word_embedding(t, glove) for t in texts],
    }

# ── NLI labels ────────────────────────────────────────────────────────────────
nli_df = pd.read_csv(NLI)
myth_nli = nli_df[nli_df["myth_category"] == "MYTH"]
nli_labels = (
    myth_nli.groupby("narrative_index")
    .apply(lambda g: dict(zip(g["myth_type"], g["overall_label"])))
    .to_dict()
)

def is_neutral(idx, myth_type):
    return nli_labels.get(idx, {}).get(myth_type, "neutral") == "neutral"


for model in MODELS:
    # ── Load ──────────────────────────────────────────────────────────────────────
    t1_path = Path(args.t1) if args.t1 else BASE / f"{model}_advice_t1.csv"
    t2_path = Path(args.t2) if args.t2 else BASE / f"{model}_advice_t2.csv"

    t1_df = pd.read_csv(t1_path)
    t2_df = pd.read_csv(t2_path, on_bad_lines="skip", engine="python")
    
    # ── T1: attach NLI labels ─────────────────────────────────────────────────────
    t1_df["narrative_nli_label"] = t1_df.apply(
        lambda r: {mt: nli_labels.get(int(r["narrative_idx"]), {}).get(mt, "neutral")
                   for mt in ["clothing", "victim_intoxication", "perpetrator_intoxication", "resistance"]},
        axis=1
    ).astype(str)
    
    # ── Build T2 paired rows (neutral only) ───────────────────────────────────────
    t1_index = t1_df.set_index(["narrative_idx", "model", "prompt_variant"])
    
    paired = []
    for _, r2 in t2_df.iterrows():
        idx       = int(r2["narrative_idx"])
        myth_type = r2.get("myth_type")
        myth_pair = r2.get("myth_pair")
    
        # neutrality filter
        if pd.notna(myth_type) and not is_neutral(idx, myth_type): # for single myths
            continue
        if pd.notna(myth_pair) and not all(is_neutral(idx, m) for m in str(myth_pair).split("+")): # for paired myths
            continue
    
        key = (idx, r2["model"], r2["prompt_variant"])
        if key not in t1_index.index:
            raise ValueError(f"T2 row has no matching T1: {key}")
        t1_row = t1_index.loc[key]
        if isinstance(t1_row, pd.DataFrame):
            t1_row = t1_row.iloc[0]
    
        paired.append({
            "narrative_idx":  idx,
            "model":          r2["model"],
            "prompt_variant": r2["prompt_variant"],
            "myth_type":      myth_type if pd.notna(myth_type) else None,
            "myth_pair":      myth_pair if pd.notna(myth_pair) else None,
            "frame":          r2["frame"],
            "dose":           r2["dose"],
            "response_t1":    str(t1_row["response_t1"]),
            "response_t2":    str(r2["response_t2"]),
        })
    
    t2_paired = pd.DataFrame(paired)
    print(f"T1: {len(t1_df)} rows | T2 paired: {len(t2_paired)} rows")
    
    # ── Embeddings ────────────────────────────────────────────────────────────────
    t1_vecs = embed_all(t1_df["response_t1"].tolist())
    t2_vecs = embed_all(t2_paired["response_t2"].tolist())
    # t1 vectors aligned to t2_paired rows (matched via index)
    t2_t1_vecs = embed_all(t2_paired["response_t1"].tolist())
    
    # ── Save PKLs ─────────────────────────────────────────────────────────────────
    for emb in EMBS:
        pd.Series(t1_vecs[emb],   name=f"{emb}_t1").to_pickle(OUT_DIR / emb / f"{model}_t1.pkl")
        pd.DataFrame({
            f"{emb}_t1": t2_t1_vecs[emb],
            f"{emb}_t2": t2_vecs[emb],
        }).to_pickle(OUT_DIR / emb / f"{model}_t2_pairs.pkl")
    
    # ── Save metadata ─────────────────────────────────────────────────────────────
    t1_df[["narrative_idx", "model", "prompt_variant", "narrative_nli_label", "response_t1"]].to_csv(
        OUT_DIR / f"{model}_t1_metadata.csv", index=False)
    
    t2_paired.to_csv(OUT_DIR / f"{model}_t2_metadata.csv", index=False)
    print("Done.")