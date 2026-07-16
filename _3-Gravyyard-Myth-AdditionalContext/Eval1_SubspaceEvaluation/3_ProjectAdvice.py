"""
project_advice.py
Encodes all advice responses and saves projection scores onto myth subspace.
"""

import os, glob, pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

os.environ['HF_HOME'] = '/mnt/beegfs/msaxena4/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/beegfs/msaxena4/hf_cache'

ADVICE_DIR    = "/mnt/beegfs/msaxena4/2_CommonMyths/B_CSCW2026/Exp1_AdviceGeneration/SampleResults"
SUBSPACE_PATH = "myth_subspace.pkl"
OUTPUT_CSV    = "Results/AllAdvice_Projected.csv"

os.makedirs("Results", exist_ok=True)

# Load
with open(SUBSPACE_PATH, "rb") as f:
    subspace_vector = pickle.load(f)
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Load all advice CSVs
files = glob.glob(f"{ADVICE_DIR}/*_advice_*.csv")
print(f"Found {len(files)} files")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"Total rows: {len(df)}")

# Filter refusals and empty responses
df = df[df["suspected_refusal"] != True]
df = df[df["response"].notna() & (df["response"].str.strip() != "")]
df = df.reset_index(drop=True)
print(f"After filtering: {len(df)} rows")

# Project
print("Projecting...")
vecs = encoder.encode(df["response"].tolist(), normalize_embeddings=True,
                      show_progress_bar=True, batch_size=128)
df["projection_score"] = np.dot(vecs, subspace_vector)

# Save
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}")