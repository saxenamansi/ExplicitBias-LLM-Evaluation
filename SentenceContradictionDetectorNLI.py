"""
Sentence-level NLI Contradiction Detector
Detects contradictions by checking against individual sentences in the narrative.
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
        entailment_detected = len(entailing_sentences) > 0  
        
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
            'entailment_detected': entailment_detected, 
            'label': overall_label,
            'confidence_score': max_contradiction_score,
            'entailment_confidence_score': max_entailment_score,
            'total_sentences': len(sentences),
            'contradicting_sentences_count': len(contradicting_sentences),
            'entailing_sentences_count': len(entailing_sentences),
            'contradicting_sentences': contradicting_sentences,
            'entailing_sentences': entailing_sentences, 
        }
        
        if return_all_sentences:
            final_result['all_sentence_results'] = sentence_results
        
        if verbose:
            status = "CONTRADICTION" if contradiction_detected else "ENTAILMENT" if entailment_detected else "NEUTRAL"
            print(f"  → {status} (contra: {max_contradiction_score:.3f}, entail: {max_entailment_score:.3f}, "
                  f"{len(contradicting_sentences)} contradict, {len(entailing_sentences)} entail)")
        
        return final_result
