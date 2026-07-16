"""
Sentence-level NLI Contradiction Detector
Detects contradictions by checking against individual sentences in the narrative.
No chunking - processes sentence-by-sentence for precision.
"""

import torch
import re
import pandas as pd
import numpy as np
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.tokenize import sent_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def create_stratified_sample(df, text_column='Content', n_samples=100, n_bins=10, random_state=42):
    """
    Create a stratified sample based on text length distribution.
    Ensures all narrative lengths are represented proportionally.
    
    Args:
        df: DataFrame with narratives
        text_column: Column name containing text
        n_samples: Total samples to draw
        n_bins: Number of length bins (more bins = finer stratification)
        random_state: Random seed for reproducibility
    
    Returns:
        Sampled DataFrame with representative length distribution
    """
    
    print(f"\n{'='*80}")
    print("CREATING STRATIFIED SAMPLE")
    print(f"{'='*80}")
    
    # Calculate text lengths
    df_copy = df.copy()
    df_copy['text_length'] = df_copy[text_column].apply(len)
    
    print(f"\nOriginal dataset: {len(df_copy)} narratives")
    print(f"Length statistics:")
    print(df_copy['text_length'].describe())
    
    # Create bins based on quantiles
    df_copy['length_bin'] = pd.qcut(df_copy['text_length'], q=n_bins, labels=False, duplicates='drop')
    
    # Calculate samples per bin (proportional to bin size)
    bin_counts = df_copy['length_bin'].value_counts().sort_index()
    samples_per_bin = (bin_counts / len(df_copy) * n_samples).round().astype(int)
    
    print(f"\nBin distribution:")
    for bin_idx in sorted(bin_counts.index):
        bin_data = df_copy[df_copy['length_bin'] == bin_idx]
        min_len = bin_data['text_length'].min()
        max_len = bin_data['text_length'].max()
        n_in_bin = bin_counts[bin_idx]
        n_to_sample = samples_per_bin[bin_idx]
        print(f"  Bin {bin_idx}: {min_len:,}-{max_len:,} chars | {n_in_bin} narratives → {n_to_sample} samples")
    
    # Adjust to ensure exactly n_samples
    diff = n_samples - samples_per_bin.sum()
    if diff > 0:
        # Add extra samples to largest bins
        largest_bins = bin_counts.nlargest(diff).index
        for i, bin_idx in enumerate(largest_bins):
            if i < diff:
                samples_per_bin[bin_idx] += 1
    elif diff < 0:
        # Remove samples from largest bins
        largest_bins = bin_counts.nlargest(abs(diff)).index
        for i, bin_idx in enumerate(largest_bins):
            if i < abs(diff):
                samples_per_bin[bin_idx] = max(0, samples_per_bin[bin_idx] - 1)
    
    # Sample from each bin
    sampled_dfs = []
    np.random.seed(random_state)
    
    for bin_idx in sorted(samples_per_bin.index):
        n = samples_per_bin[bin_idx]
        if n == 0:
            continue
        
        bin_data = df_copy[df_copy['length_bin'] == bin_idx]
        if len(bin_data) >= n:
            sampled = bin_data.sample(n=n, random_state=random_state)
        else:
            sampled = bin_data  # Take all if bin is too small
        sampled_dfs.append(sampled)
    
    # Combine
    result = pd.concat(sampled_dfs)
    
    print(f"\n{'='*80}")
    print(f"SAMPLED {len(result)} NARRATIVES")
    print(f"{'='*80}")
    print(f"\nSampled length distribution:")
    print(result['text_length'].describe())
    
    # Drop helper columns
    result = result.drop(columns=['length_bin', 'text_length'])
    
    return result


class SentenceLevelContradictionDetector:
    """
    Detects contradictions by checking myth sentence against each sentence in narrative.
    Returns contradiction if ANY sentence contradicts.
    """
    
    def __init__(self, model_name: str, contradiction_threshold: float = 0.5):
        """
        Initialize the contradiction detector.
        
        Args:
            model_name: HuggingFace model identifier for NLI
            contradiction_threshold: Minimum probability to consider as contradiction
        """
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.contradiction_threshold = contradiction_threshold
        
        print(f"Model loaded on {self.device}")
        
        # Label mapping for the model
        self.id2label = self.model.config.id2label
    
    def preprocess_text(self, text: str) -> str:
        """Clean Reddit-specific formatting from text."""
        # Remove Reddit markdown
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        text = re.sub(r'~~(.+?)~~', r'\1', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove user mentions
        text = re.sub(r'u/[A-Za-z0-9_-]+', '', text)
        text = re.sub(r'/u/[A-Za-z0-9_-]+', '', text)
        
        # Remove subreddit mentions
        text = re.sub(r'r/[A-Za-z0-9_-]+', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split narrative into individual sentences."""
        sentences = sent_tokenize(text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences
    
    def infer_nli(self, premise: str, hypothesis: str) -> Dict[str, any]:
        """Perform NLI inference on a premise-hypothesis pair."""
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]
        
        label_scores = {
            self.id2label[i]: probs[i].item() 
            for i in range(len(probs))
        }
        
        predicted_label = max(label_scores, key=label_scores.get)
        
        return {
            'label': predicted_label,
            'scores': label_scores,
            'contradiction_score': label_scores.get('CONTRADICTION', label_scores.get('contradiction', 0.0))
        }
    
    def detect_contradiction(self, myth_sentence: str, narrative: str, 
                           return_all_sentences: bool = False,
                           verbose: bool = True) -> Dict[str, any]:
        """
        Detect contradictions sentence-by-sentence.
        Returns contradiction if ANY sentence contradicts.
        """
        
        clean_myth = self.preprocess_text(myth_sentence)
        clean_narrative = self.preprocess_text(narrative)
        
        sentences = self.split_into_sentences(clean_narrative)
        
        if verbose:
            print(f"  Analyzing {len(sentences)} sentences")
        
        sentence_results = []
        contradicting_sentences = []  # Track contradictions
        entailing_sentences = []  # Track entailments
        
        for i, sentence in enumerate(sentences):
            result = self.infer_nli(sentence, clean_myth)
            
            sentence_results.append({
                'sentence': sentence,
                'sentence_index': i,
                'result': result
            })
            
            # Track contradictions
            if result['contradiction_score'] >= self.contradiction_threshold:
                contradicting_sentences.append({
                    'sentence': sentence,
                    'sentence_index': i,
                    'contradiction_score': result['contradiction_score'],
                    'label_scores': result['scores']
                })
            
            # Track entailments
            entailment_score = result['scores'].get('ENTAILMENT', result['scores'].get('entailment', 0.0))
            if entailment_score >= self.contradiction_threshold:  # Use same threshold
                entailing_sentences.append({
                    'sentence': sentence,
                    'sentence_index': i,
                    'entailment_score': entailment_score,
                    'label_scores': result['scores']
                })
        
        # Sort by score (highest first)
        contradicting_sentences.sort(key=lambda x: x['contradiction_score'], reverse=True)
        entailing_sentences.sort(key=lambda x: x['entailment_score'], reverse=True)
        
        # Contradiction if ANY sentence contradicts, Entailment if ANY sentence entails
        contradiction_detected = len(contradicting_sentences) > 0
        entailment_detected = len(entailing_sentences) > 0  # NEW
        
        if sentence_results:
            max_contradiction_score = max([r['result']['contradiction_score'] for r in sentence_results])
            max_entailment_score = max([
                r['result']['scores'].get('ENTAILMENT', r['result']['scores'].get('entailment', 0.0))
                for r in sentence_results
            ])
        else:
            max_contradiction_score = 0.0
            max_entailment_score = 0.0
        
        # Determine overall label (priority: contradiction > entailment > neutral)
        if contradiction_detected:
            overall_label = 'contradiction'
        elif entailment_detected:
            overall_label = 'entailment'
        else:
            overall_label = 'neutral'
        
        final_result = {
            'contradiction_detected': contradiction_detected,
            'entailment_detected': entailment_detected,  # NEW
            'label': overall_label,
            'confidence_score': max_contradiction_score,
            'entailment_confidence_score': max_entailment_score,  # NEW
            'total_sentences': len(sentences),
            'contradicting_sentences_count': len(contradicting_sentences),
            'entailing_sentences_count': len(entailing_sentences),  # NEW
            'contradicting_sentences': contradicting_sentences,
            'entailing_sentences': entailing_sentences,  # NEW
        }
        
        if return_all_sentences:
            final_result['all_sentence_results'] = sentence_results
        
        if verbose:
            status = "CONTRADICTION" if contradiction_detected else "ENTAILMENT" if entailment_detected else "NEUTRAL"
            print(f"  → {status} (contra: {max_contradiction_score:.3f}, entail: {max_entailment_score:.3f}, "
                  f"{len(contradicting_sentences)} contradict, {len(entailing_sentences)} entail)")
        
        return final_result
