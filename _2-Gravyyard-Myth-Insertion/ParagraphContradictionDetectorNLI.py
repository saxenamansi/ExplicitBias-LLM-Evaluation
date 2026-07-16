import os, sys
# Cache paths
os.environ['HF_HOME'] = '/mnt/beegfs/msaxena4/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/beegfs/msaxena4/hf_cache'

# Test write permission
test_file_path = os.path.join(os.environ['TRANSFORMERS_CACHE'], 'text.txt')
try:
    with open(test_file_path, 'w') as f:
        f.write('This is a test.')
    print('Write successful!')
except Exception as e:
    print('Error writing to directory: ', e)

from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN"))

"""
NLI-based Contradiction Detector
Detects if a sentence contradicts any part of a long narrative using semantic chunking.
"""

import time
import numpy as np
import pandas as pd
import torch
import re
from typing import List, Dict, Tuple
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


class ContradictionDetector:
    """
    Detects contradictions between a sentence and a long narrative using NLI.
    Uses semantic chunking based on paragraphs to handle long Reddit posts.
    """
    
    def __init__(self, model_name: str,
                 max_chunk_tokens: int = 400,
                 contradiction_threshold: float = 0.5):
        """
        Initialize the contradiction detector.
        
        Args:
            model_name: HuggingFace model identifier for NLI
            max_chunk_tokens: Maximum tokens per chunk
            contradiction_threshold: Minimum probability to consider as contradiction
        """
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.max_chunk_tokens = max_chunk_tokens
        self.contradiction_threshold = contradiction_threshold
        
        print(f"Model loaded on {self.device}")
        
        # Label mapping for the model
        self.id2label = self.model.config.id2label
        
    def preprocess_text(self, text: str) -> str:
        """
        Clean Reddit-specific formatting from text.
        
        Args:
            text: Raw text from Reddit post
            
        Returns:
            Cleaned text
        """
        # Remove Reddit markdown
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'__(.+?)__', r'\1', text)      # Bold alternative
        text = re.sub(r'\*(.+?)\*', r'\1', text)      # Italic
        text = re.sub(r'_(.+?)_', r'\1', text)        # Italic alternative
        text = re.sub(r'~~(.+?)~~', r'\1', text)      # Strikethrough
        
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
    
    def chunk_narrative_semantic(self, text: str) -> List[Dict[str, any]]:
        """
        Chunk narrative semantically based on paragraphs, respecting token limits.
        
        Args:
            text: Preprocessed narrative text
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        # Split by paragraphs (double newline or single newline)
        paragraphs = [p.strip() for p in re.split(r'\n+', text) if p.strip()]
        
        if not paragraphs:
            # If no paragraphs, split by sentences
            paragraphs = sent_tokenize(text)
        
        current_chunk = ""
        current_sentences = []
        chunk_index = 0
        
        for para in paragraphs:
            # Tokenize the paragraph to check length
            para_tokens = self.tokenizer.tokenize(para)
            current_tokens = self.tokenizer.tokenize(current_chunk)
            
            # If adding this paragraph exceeds max tokens, save current chunk
            if len(current_tokens) + len(para_tokens) > self.max_chunk_tokens and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'chunk_index': chunk_index,
                    'token_count': len(current_tokens)
                })
                current_chunk = ""
                current_sentences = []
                chunk_index += 1
            
            # If single paragraph is too long, split by sentences
            if len(para_tokens) > self.max_chunk_tokens:
                sentences = sent_tokenize(para)
                for sent in sentences:
                    sent_tokens = self.tokenizer.tokenize(sent)
                    current_tokens = self.tokenizer.tokenize(current_chunk)
                    
                    if len(current_tokens) + len(sent_tokens) > self.max_chunk_tokens and current_chunk:
                        chunks.append({
                            'text': current_chunk.strip(),
                            'chunk_index': chunk_index,
                            'token_count': len(current_tokens)
                        })
                        current_chunk = ""
                        chunk_index += 1
                    
                    current_chunk += " " + sent
            else:
                current_chunk += " " + para
        
        # Add remaining chunk
        if current_chunk.strip():
            current_tokens = self.tokenizer.tokenize(current_chunk)
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_index': chunk_index,
                'token_count': len(current_tokens)
            })
        
        return chunks
    
    def infer_nli(self, premise: str, hypothesis: str) -> Dict[str, any]:
        """
        Perform NLI inference on a premise-hypothesis pair.
        
        Args:
            premise: The context (narrative chunk)
            hypothesis: The sentence to check
            
        Returns:
            Dictionary with label and probability scores
        """
        # Tokenize input
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]
        
        # Map to labels
        label_scores = {
            self.id2label[i]: probs[i].item() 
            for i in range(len(probs))
        }
        
        # Get predicted label
        predicted_label = max(label_scores, key=label_scores.get)
        
        return {
            'label': predicted_label,
            'scores': label_scores,
            'contradiction_score': label_scores.get('CONTRADICTION', label_scores.get('contradiction', 0.0))
        }
    
    def aggregate_results(self, chunk_results: List[Dict]) -> Dict[str, any]:
        """
        Aggregate NLI results across all chunks.
        
        Args:
            chunk_results: List of results from each chunk
            
        Returns:
            Aggregated results with overall decision
        """
        if not chunk_results:
            return {
                'contradiction_detected': False,
                'confidence_score': 0.0,
                'contradicting_segments': [],
                'label': 'neutral'
            }
        
        # Find chunks with contradiction
        contradicting_chunks = [
            r for r in chunk_results 
            if r['result']['contradiction_score'] >= self.contradiction_threshold
        ]
        
        # Sort by contradiction score
        contradicting_chunks.sort(key=lambda x: x['result']['contradiction_score'], reverse=True)
        
        # Overall decision: contradiction if any chunk contradicts
        contradiction_detected = len(contradicting_chunks) > 0
        
        # Confidence is the max contradiction score
        max_contradiction_score = max(
            [r['result']['contradiction_score'] for r in chunk_results]
        ) if chunk_results else 0.0
        
        # Prepare contradicting segments
        contradicting_segments = [
            {
                'text': chunk['chunk']['text'][:200] + '...' if len(chunk['chunk']['text']) > 200 else chunk['chunk']['text'],
                'full_text': chunk['chunk']['text'],
                'score': chunk['result']['contradiction_score'],
                'chunk_index': chunk['chunk']['chunk_index'],
                'label_scores': chunk['result']['scores']
            }
            for chunk in contradicting_chunks
        ]
        
        # Determine overall label
        if contradiction_detected:
            overall_label = 'contradiction'
        else:
            # Check if any entailment
            max_entailment = max([
                r['result']['scores'].get('ENTAILMENT', r['result']['scores'].get('entailment', 0.0))
                for r in chunk_results
            ])
            overall_label = 'entailment' if max_entailment > 0.5 else 'neutral'
        
        return {
            'contradiction_detected': contradiction_detected,
            'confidence_score': max_contradiction_score,
            'contradicting_segments': contradicting_segments,
            'label': overall_label,
            'total_chunks': len(chunk_results),
            'contradicting_chunks_count': len(contradicting_chunks)
        }
    
    def detect_contradiction(self, sentence: str, narrative: str, 
                           return_all_chunks: bool = False,
                           verbose: bool = True) -> Dict[str, any]:
        """
        Main pipeline to detect contradictions.
        
        Args:
            sentence: The sentence to check for contradictions
            narrative: The long narrative (Reddit post)
            return_all_chunks: Whether to return detailed results for all chunks
            verbose: Whether to print progress information
            
        Returns:
            Dictionary with contradiction detection results
        """
        import time
        
        start_time = time.time()
        
        # Preprocess
        clean_sentence = self.preprocess_text(sentence)
        clean_narrative = self.preprocess_text(narrative)
        
        # Chunk the narrative
        chunks = self.chunk_narrative_semantic(clean_narrative)
        
        if verbose:
            print(f"  Created {len(chunks)} chunks from narrative ({len(narrative)} chars)")
        
        # Run NLI on each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks):
            result = self.infer_nli(chunk['text'], clean_sentence)
            chunk_results.append({
                'chunk': chunk,
                'result': result
            })
            
            # Print progress for long narratives
            if verbose and len(chunks) > 10:
                if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                    elapsed = time.time() - start_time
                    chunks_per_sec = (i + 1) / elapsed
                    remaining = (len(chunks) - (i + 1)) / chunks_per_sec if chunks_per_sec > 0 else 0
                    print(f"    Progress: {i + 1}/{len(chunks)} chunks ({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")
        
        # Aggregate results
        final_result = self.aggregate_results(chunk_results)
        
        # Add timing info
        total_time = time.time() - start_time
        final_result['processing_time_seconds'] = total_time
        
        # Add all chunk results if requested
        if return_all_chunks:
            final_result['all_chunk_results'] = chunk_results
        
        if verbose:
            print(f"  Completed in {total_time:.2f}s | Result: {final_result['label']} (confidence: {final_result['confidence_score']:.3f})")
        
        return final_result
    
    def print_results(self, results: Dict[str, any]):
        """
        Pretty print the results.
        
        Args:
            results: Results dictionary from detect_contradiction
        """
        print("\n" + "="*80)
        print("CONTRADICTION DETECTION RESULTS")
        print("="*80)
        
        print(f"\nContradiction Detected: {results['contradiction_detected']}")
        print(f"Confidence Score: {results['confidence_score']:.3f}")
        print(f"Overall Label: {results['label']}")
        print(f"Total Chunks Analyzed: {results['total_chunks']}")
        
        if results['contradicting_segments']:
            print(f"\nFound {len(results['contradicting_segments'])} contradicting segment(s):")
            print("-" * 80)
            
            for i, segment in enumerate(results['contradicting_segments'][:3], 1):  # Show top 3
                print(f"\nSegment {i} (Score: {segment['score']:.3f}):")
                print(f"  {segment['text']}")
                print(f"  Label scores: {segment['label_scores']}")
        else:
            print("\nNo contradictions found.")
        
        print("\n" + "="*80)