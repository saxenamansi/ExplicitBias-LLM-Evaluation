"""Encode real stories T1/T2 responses and save embeddings in Reddit pipeline format."""
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

SOURCES = {
    "JustDetention":   Path("Results/JustDetention"),
    "SurvivorStories": Path("Results/SurvivorStories"),
}
MODELS = ["gemma", "llama", "mistral", "qwen"]
TASKS  = ["advice", "summary"]

sbert = SentenceTransformer("all-mpnet-base-v2")

def encode(texts):
    return list(sbert.encode(texts, normalize_embeddings=True))

for source, base_dir in SOURCES.items():
    emb_dir  = base_dir / "1_Embeddings" / "SBERT"
    meta_dir = base_dir / "1_Embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        for model in MODELS:
            t1_df = pd.read_csv(base_dir / f"{model}_{task}_t1.csv")
            t2_df = pd.read_csv(base_dir / f"{model}_{task}_t2.csv")

            # ── Build T2 paired rows ──────────────────────────────────────────
            t1_index = t1_df.set_index("narrative_idx")
            paired = []
            for _, r2 in t2_df.iterrows():
                idx = int(r2["narrative_idx"])
                if idx not in t1_index.index:
                    raise ValueError(f"T2 row has no matching T1: {idx}")
                t1_row = t1_index.loc[idx]
                if isinstance(t1_row, pd.DataFrame):
                    t1_row = t1_row.iloc[0]
                paired.append({
                    "narrative_idx":  idx,
                    "model":          r2["model"],
                    "prompt_variant": r2.get("prompt_variant", "default"),
                    "myth_type":      r2.get("myth_type"),
                    "myth_pair":      r2.get("myth_pair"),
                    "frame":          r2["frame"],
                    "dose":           r2["dose"],
                    "response_t1":    str(t1_row["response_t1"]),
                    "response_t2":    str(r2["response_t2"]),
                })
            t2_paired = pd.DataFrame(paired)

            # ── Embeddings ────────────────────────────────────────────────────
            t1_vecs    = encode(t1_df["response_t1"].tolist())
            t2_vecs    = encode(t2_paired["response_t2"].tolist())
            t2_t1_vecs = encode(t2_paired["response_t1"].tolist())

            # ── Save PKLs ─────────────────────────────────────────────────────
            pd.Series(t1_vecs, name="SBERT_t1").to_pickle(emb_dir / f"{model}_{task}_t1.pkl")
            pd.DataFrame({
                "SBERT_t1": t2_t1_vecs,
                "SBERT_t2": t2_vecs,
            }).to_pickle(emb_dir / f"{model}_{task}_t2_pairs.pkl")

            if "prompt_variant" not in t1_df.columns:
                t1_df["prompt_variant"] = "default"
            t1_df[["narrative_idx", "model", "prompt_variant", "nli_labels", "response_t1"]].rename(
            columns={"nli_labels": "narrative_nli_label"}).to_csv(
            meta_dir / f"{model}_{task}_t1_metadata.csv", index=False)

            t2_paired.to_csv(meta_dir / f"{model}_{task}_t2_metadata.csv", index=False)
            
            print(f"{source} {task} {model}: T1={len(t1_vecs)} T2={len(t2_vecs)}")