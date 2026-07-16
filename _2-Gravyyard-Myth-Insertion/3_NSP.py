import os, sys

# Cache paths
os.environ['HF_HOME'] = '/mnt/beegfs/msaxena4/hf_cache'
print("Cache directories set.")
# Test write permission
test_file_path = os.path.join(os.environ['HF_HOME'], 'text.txt')
try:
    with open(test_file_path, 'w') as f:
        print('Cache write test successful.')
except Exception as e:
    print('Error writing to cache directory:', e)

import pandas as pd
import numpy as np
import random
import math
import ast
import torch
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer, BertForNextSentencePrediction

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)
print("Model ready.")

def get_NSP_prob(model, tokenizer, s1, s2):
    """Returns (prob, skipped). skipped=True if sequence exceeded 512 tokens."""
    encoding = tokenizer(s1, s2, return_tensors='pt')
    if encoding['input_ids'].shape[1] > 512:
        return float('nan'), True
    encoding = {k: v.to(model.device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.softmax(outputs.logits, dim=1)
    return probs[0, 0].item(), False

def get_each_candidate_NSP(row, model, tokenizer):
    sentences = row["modified_sentences"]
    myth_idx = int(row["myth_sentence_idx"])
    prev_idx = row["prev_sentence_idx"]
    next_idx = row["next_sentence_idx"]
    prev_prob = float('nan')
    next_prob = float('nan')
    orig_prob = float('nan')
    prev_skipped = False
    next_skipped = False
    orig_skipped = False

    if not math.isnan(prev_idx):
        prev_idx = int(prev_idx)
        prev_prob, prev_skipped = get_NSP_prob(
            model, tokenizer,
            sentences[prev_idx],
            sentences[myth_idx]
        )
    if not math.isnan(next_idx):
        next_idx = int(next_idx)
        next_prob, next_skipped = get_NSP_prob(
            model, tokenizer,
            sentences[myth_idx],
            sentences[next_idx]
        )
    if not math.isnan(prev_idx) and not math.isnan(next_idx):
        orig_prob, orig_skipped = get_NSP_prob(
            model, tokenizer,
            sentences[int(prev_idx)],
            sentences[int(next_idx)]
        )
    return prev_prob, next_prob, orig_prob, prev_skipped, next_skipped, orig_skipped

def get_all_candidate_NSP(df):
    results = []
    total = len(df)
    skipped_counts = {"prev": 0, "next": 0, "orig": 0}
    print(f"Processing {total} rows...")
    for idx, row in df.iterrows():
        if idx % 50 == 0:
            print(f"  → Processing row {idx+1}/{total}")
        prev_prob, next_prob, orig_prob, prev_skipped, next_skipped, orig_skipped = get_each_candidate_NSP(
            row, model, tokenizer
        )
        if prev_skipped: skipped_counts["prev"] += 1
        if next_skipped: skipped_counts["next"] += 1
        if orig_skipped: skipped_counts["orig"] += 1
        results.append({
            "narrative_idx": row["narrative_idx"],
            "original_narrative": row["original_narrative"],
            "myth_type": row["myth_type"],
            "myth_variation": row["myth_variation"],
            "dose": row["dose"],
            "myth_detail": row["myth_detail"],
            "modified_narrative": row["modified_narrative"],
            "modified_sentences": row["modified_sentences"],
            "insertion_idx": row["myth_sentence_idx"],
            "prev_prob": prev_prob,
            "next_prob": next_prob,
            "orig_prob": orig_prob,
            "prev_skipped": prev_skipped,
            "next_skipped": next_skipped,
            "orig_skipped": orig_skipped
        })
    print("Finished NSP computation.")
    print(f"Skipped due to >512 tokens — prev: {skipped_counts['prev']}, next: {skipped_counts['next']}, orig: {skipped_counts['orig']}")
    return pd.DataFrame(results)

reddit_df = pd.read_csv(
    'Results/ModifiedNarrativeCandidates-(SV)-all.csv',
    converters={"modified_sentences": ast.literal_eval}
)
df_out = get_all_candidate_NSP(reddit_df)
df_out.to_csv("Results/Reddit_candidates-(SV)-NSP-all.csv", index=False)
print("Done.")