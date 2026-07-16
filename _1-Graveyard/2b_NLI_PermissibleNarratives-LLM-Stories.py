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

"""
NLI-based Contradiction Detector for Reddit Posts
Detects if a sentence contradicts any part of a long narrative using semantic chunking.
"""

import os
import torch
import re
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize
import warnings
warnings.filterwarnings('ignore')

from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN"))

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ContradictionDetector:
    """
    Detects contradictions between a sentence and a long narrative using NLI.
    Uses semantic chunking based on paragraphs to handle long Reddit posts.
    """
    
    def __init__(self, model_name: str = "microsoft/deberta-v3-base-mnli", 
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
                           return_all_chunks: bool = False) -> Dict[str, any]:
        """
        Main pipeline to detect contradictions.
        
        Args:
            sentence: The sentence to check for contradictions
            narrative: The long narrative (Reddit post)
            return_all_chunks: Whether to return detailed results for all chunks
            
        Returns:
            Dictionary with contradiction detection results
        """
        # Preprocess
        clean_sentence = self.preprocess_text(sentence)
        clean_narrative = self.preprocess_text(narrative)
        
        # Chunk the narrative
        chunks = self.chunk_narrative_semantic(clean_narrative)
        
        print(f"Processing {len(chunks)} chunks...")
        
        # Run NLI on each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks):
            result = self.infer_nli(chunk['text'], clean_sentence)
            chunk_results.append({
                'chunk': chunk,
                'result': result
            })
            
            # Print progress
            if (i + 1) % 5 == 0 or (i + 1) == len(chunks):
                print(f"  Processed {i + 1}/{len(chunks)} chunks")
        
        # Aggregate results
        final_result = self.aggregate_results(chunk_results)
        
        # Add all chunk results if requested
        if return_all_chunks:
            final_result['all_chunk_results'] = chunk_results
        
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

def process_dataset(df, detector):
    # Process each row
    print("\nProcessing rows...")
    results_list = []
    
    for idx, row in df.iterrows():
        print(f"\nProcessing row {idx + 1}/{len(df)}...")
        narrative = row['OriginalNarrative']

        for j, myth_variations in enumerate([myth_variations_d1, myth_variations_d2]):
            for myth_var, myth_details in myth_variations.items():
                for myth_type, myth_detail in myth_details.items():
        
                    # Run contradiction detection
                    result = detector.detect_contradiction(myth_detail, narrative, return_all_chunks=True)
                    
                    # Extract all relevant information
                    row_result = {
                        'row_index': idx,
                        'narrative': narrative,
                        "myth_type": myth_type,
                        "myth_variation" : myth_var,
                        "dose" : j+1,
                        "myth_detail": myth_detail,
                        'contradiction_detected': result['contradiction_detected'],
                        'overall_label': result['label'],
                        'confidence_score': result['confidence_score'],
                        'total_chunks': result['total_chunks'],
                        'contradicting_chunks_count': result['contradicting_chunks_count'],
                    }
        
                    # Add probabilities from the highest-scoring chunk
                    if result.get('all_chunk_results'):
                        # Get the chunk with max contradiction score
                        max_chunk = max(result['all_chunk_results'], 
                                      key=lambda x: x['result']['contradiction_score'])
                        
                        row_result['max_contradiction_prob'] = max_chunk['result']['scores'].get(
                            'CONTRADICTION', max_chunk['result']['scores'].get('contradiction', 0.0)
                        )
                        row_result['max_neutral_prob'] = max_chunk['result']['scores'].get(
                            'NEUTRAL', max_chunk['result']['scores'].get('neutral', 0.0)
                        )
                        row_result['max_entailment_prob'] = max_chunk['result']['scores'].get(
                            'ENTAILMENT', max_chunk['result']['scores'].get('entailment', 0.0)
                        )
                        
                        # Add contradicting segment text if exists
                        if result['contradicting_segments']:
                            row_result['top_contradicting_segment'] = result['contradicting_segments'][0]['full_text']
                            row_result['top_contradicting_score'] = result['contradicting_segments'][0]['score']
                        else:
                            row_result['top_contradicting_segment'] = None
                            row_result['top_contradicting_score'] = None
                        
                        # Store all chunk-level results as JSON string
                        chunk_details = []
                        for chunk_result in result['all_chunk_results']:
                            chunk_details.append({
                                'chunk_index': chunk_result['chunk']['chunk_index'],
                                'chunk_text': chunk_result['chunk']['text'][:100] + '...',  # First 100 chars
                                'label': chunk_result['result']['label'],
                                'contradiction_prob': chunk_result['result']['contradiction_score'],
                                'scores': chunk_result['result']['scores']
                            })
                        
                        row_result['all_chunk_details'] = str(chunk_details)  # Store as string for CSV compatibility
                    
                    results_list.append(row_result)
        
        # Print summary for this row
        print(f"  Result: {result['label']} (confidence: {result['confidence_score']:.3f})")
    
    # Create results dataframe
    results_df = pd.DataFrame(results_list)
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print(f"Total rows processed: {len(results_df)}")
    print(f"Contradictions detected: {results_df['contradiction_detected'].sum()}")
    print(f"Non-contradictions: {(~results_df['contradiction_detected']).sum()}")
    print(f"Average confidence score: {results_df['confidence_score'].mean():.3f}")
    
    print("\nLabel distribution:")
    print(results_df['overall_label'].value_counts())
    
    return results_df

NEG_MYTH_d1 = {
    "clothing" : "I was not wearing a revealing outfit.",
    "victim_intoxication" : "I was not intoxicated.",
    "perpetrator_intoxication" :  "The perpetrator was not intoxicated.",
    "resistance" : "I resisted the perpetrator.",
}

NEG_NON_MYTH_d1 = {
    "clothing" : "I was not wearing a modest outfit.",
    "victim_intoxication" :  "I was not sober.",
    "perpetrator_intoxication" :  "The perpetrator was not sober.",
    "resistance" : "I yielded to the perpetrator."    
}

POS_MYTH_d1 = {
    "clothing" : "I was wearing a revealing outfit.", 
    "victim_intoxication" : "I was intoxicated.",
    "perpetrator_intoxication" : "The perpetrator was intoxicated.",
    "resistance" : "I did not resist the perpetrator.",
}

POS_NON_MYTH_d1 = {
    "clothing" : "I was wearing a modest outfit.",
    "victim_intoxication" :  "I was sober.",
    "perpetrator_intoxication" :  "The perpetrator was sober.",
    "resistance" : "I did not yield to the perpetrator.",
}

myth_variations_d1 = {
    "NegMyth" : NEG_MYTH_d1, 
    "NegNonMyth" : NEG_NON_MYTH_d1, 
    "PosMyth" : POS_MYTH_d1, 
    "PosNonMyth" : POS_NON_MYTH_d1
}

NEG_MYTH_d2 = {
    "clothing" : "I was not wearing a revealing outfit. My outfit did not expose much of my skin and body shape. Others observed that my clothing was not bold in style and did not have minimum coverage.",
    "victim_intoxication" : "I was not intoxicated. I had not indulged in consuming alcohol, and my senses were not impaired. Others observed that I did not have an unsteady walk, slurred speech, and blurred vision.",
    "perpetrator_intoxication" : "The perpetrator was not intoxicated. The perpetrator had not indulged in consuming alcohol, and their senses were not impaired. Others observed that the perpetrator did not have an unsteady walk, slurred speech, and blurred vision.",
    "resistance" : "I resisted the perpetrator. I pushed them and stepped away to prevent them from taking control. I shouted for the perpetrator to stop and hindered them from continuing."
}
    
NEG_NON_MYTH_d2 = {
    "clothing" : "I was not wearing a modest outfit. My outfit did not cover much of my skin and body shape. Others observed that my clothing was not conservative in style and did not have maximum coverage.",
    "victim_intoxication" : "I was not sober. I had not refrained from consuming alcohol, and my senses were not unaffected. Others observed that I did not have a stable walk, articulate speech, and clear vision. ",
    "perpetrator_intoxication" : "The perpetrator was not sober. The perpetrator had not refrained from consuming alcohol, and their senses were not unaffected. Others observed that the perpetrator did not have a stable walk, articulate speech, and clear vision.",
    "resistance" : "I yielded to the perpetrator. I allowed them to take control and stood still. I remained silent and let the perpetrator continue unhindered."
}

POS_MYTH_d2 = {
    "clothing" : "I was wearing a revealing outfit. My outfit exposed much of my skin and body shape. Others observed that my clothing was bold in style and had minimum coverage.",
    "victim_intoxication" : "I was intoxicated. I had indulged in consuming alcohol, and my senses were impaired. Others observed that I had an unsteady walk, slurred speech, and blurred vision. ",
    "perpetrator_intoxication" : "The perpetrator was intoxicated. The perpetrator had indulged in consuming alcohol, and their senses were impaired. Others observed that the perpetrator had an unsteady walk, slurred speech, and blurred vision.",
    "resistance" : "I did not resist the perpetrator. I did not push them and stepped away to prevent them from taking control. I did not shout for the perpetrator to stop and did not hinder them from continuing."
}

POS_NON_MYTH_d2 = {
    "clothing" : "I was wearing a modest outfit. My outfit covered much of my skin and body shape. Others observed that my clothing was conservative in style and had maximum coverage.",
    "victim_intoxication" : "I was sober. I had refrained from consuming alcohol, and my senses were unaffected. Others observed that I had a stable walk, articulate speech, and clear vision.",
    "perpetrator_intoxication" : "The perpetrator was sober. The perpetrator had refrained from consuming alcohol, and their senses were unaffected. Others observed that the perpetrator had a stable walk, articulate speech, and clear vision.",
    "resistance" : "I did not yield to the perpetrator. I did not allow them to take control and did not stay still. I did not remain silent and did not let the perpetrator continue unhindered."
}

myth_variations_d2 = {
    "NegMyth" : NEG_MYTH_d2, 
    "NegNonMyth" : NEG_NON_MYTH_d2, 
    "PosMyth" : POS_MYTH_d2, 
    "PosNonMyth" : POS_NON_MYTH_d2
}


"""
Process dataset with narrative and sentence columns.
Detects contradictions and stores all results in a dataframe.
"""

# Initialize detector
print("Initializing ContradictionDetector...")

model_name = "facebook/bart-large-mnli"

detector = ContradictionDetector(
    model_name=model_name,
    max_chunk_tokens=400,
    contradiction_threshold=0.5
)

print("\nLoading dataset...")

gemini_df = pd.read_csv('../2_GeneratingNarratives/Results/Gemini_processed.csv')
llama_df = pd.read_csv('../2_GeneratingNarratives/Results/Llama_processed.csv')
mistral_df = pd.read_csv('../2_GeneratingNarratives/Results/Mistral-100.csv')

gemini_out = process_dataset(gemini_df, detector)
llama_out = process_dataset(llama_df, detector)
llama_out = process_dataset(mistral_df, detector)

gemini_out.to_csv("Results/Gemini_NLI.csv", index=False)
llama_out.to_csv("Results/Llama_NLI.csv", index=False)
llama_out.to_csv("Results/Mistral_NLI.csv", index=False)