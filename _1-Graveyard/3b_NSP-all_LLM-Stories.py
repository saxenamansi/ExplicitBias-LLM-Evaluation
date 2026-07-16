import os, sys
# Cache paths
os.environ['HF_HOME'] = '/mnt/beegfs/msaxena4/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/beegfs/msaxena4/hf_cache'

# Test write permission
test_file_path = os.path.join(os.environ['TRANSFORMERS_CACHE'], 'text.txt')
try:
    with open(test_file_path, 'w') as f:
        print('Write successful!')
except Exception as e:
    print('Error writing to directory: ', e)

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
model = model.to(device)

NUM_NARRATIVES = 100

def get_NSP_prob(model, tokenizer, s1, s2):
    encoding = tokenizer(s1, s2, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.softmax(outputs.logits, dim=1)
    return probs[0, 0].item()   

def get_each_candidate_NSP(row, model, tokenizer):
    sentences = row["modified_sentences"]
    myth_idx = int(row["myth_sentence_idx"])

    prev_idx = row["prev_sentence_idx"]
    next_idx = row["next_sentence_idx"]

    prev_prob = float('nan')
    next_prob = float('nan')
    orig_prob = float('nan')
    
    # Previous pair
    if not math.isnan(prev_idx):
        prev_idx = int(prev_idx)
        s1 = sentences[prev_idx]
        s2 = sentences[myth_idx]
        prev_prob = get_NSP_prob(model, tokenizer, s1, s2)

    # Next pair
    if not math.isnan(next_idx):
        next_idx = int(next_idx)
        s1 = sentences[myth_idx]
        s2 = sentences[next_idx]
        next_prob = get_NSP_prob(model, tokenizer, s1, s2)

    if not math.isnan(prev_idx) and not math.isnan(next_idx):
        prev_idx = int(prev_idx)
        next_idx = int(next_idx)
        s1 = sentences[prev_idx]
        s2 = sentences[next_idx]
        orig_prob = get_NSP_prob(model, tokenizer, s1, s2)

    return prev_prob, next_prob, orig_prob

def get_all_candidate_NSP(df):
    results = []
    for idx, row in df.iterrows():
        prev_prob, next_prob, orig_prob = get_each_candidate_NSP(
            row, model, tokenizer
        )
        results.append({
            "narrative_idx": row["narrative_idx"],
            "StoryOutline": row["StoryOutline"],
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
            "orig_prob": orig_prob
        })
    
    return pd.DataFrame(results)

gemini_df = pd.read_csv('Results/Gemini_candidates.csv', converters={"modified_sentences": ast.literal_eval})
llama_df = pd.read_csv('Results/Llama_candidates.csv', converters={"modified_sentences": ast.literal_eval})
mistral_df = pd.read_csv('Results/Mistral_candidates.csv', converters={"modified_sentences": ast.literal_eval})

gemini_out = get_all_candidate_NSP(gemini_df)
llama_out = get_all_candidate_NSP(llama_df)
mistral_out = get_all_candidate_NSP(mistral_df)

gemini_out.to_csv("Results/Gemini_candidates_NSP.csv", index=False)
llama_out.to_csv("Results/Llama_candidates_NSP.csv", index=False)
mistral_out.to_csv("Results/Mistral_candidates_NSP.csv", index=False)