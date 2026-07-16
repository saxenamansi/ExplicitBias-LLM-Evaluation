"""
LLM-based Contradiction Detector using Gemini API
Alternative to NLI approach for detecting contradictions in trauma narratives.
"""

import google.generativeai as genai
import pandas as pd
import time
from typing import Dict, List
import os
import re

class GeminiContradictionDetector:
    """
    Detects contradictions using Google's Gemini API.
    Uses LLM reasoning instead of NLI models.
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the Gemini-based detector.
        
        Args:
            api_key: Google API key for Gemini
            model_name: Gemini model to use
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"temperature": 0.0}
        )
        self.model_name = model_name
        
        print(f"Initialized Gemini detector with model: {model_name}")
    
    def detect_contradiction(self, sentence: str, narrative: str, 
                           return_explanation: bool = True,
                           verbose: bool = True) -> Dict[str, any]:
        """
        Detect if sentence contradicts the narrative using Gemini.
        
        Args:
            sentence: The sentence to check
            narrative: The narrative text
            return_explanation: Whether to include LLM's explanation
            verbose: Whether to print progress
            
        Returns:
            Dictionary with contradiction detection results
        """
        start_time = time.time()
        
        # Create prompt
        prompt = self._create_prompt(sentence, narrative)
        
        if verbose:
            print(f"  Querying Gemini...")
        
        try:
            # Call Gemini API
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Parse response
            result = self._parse_response(response_text)
            
            # Add metadata
            result['processing_time_seconds'] = time.time() - start_time
            result['model_used'] = self.model_name
            result['raw_response'] = response_text if return_explanation else None
            
            if verbose:
                print(f"  Result: {result['label']} (confidence: {result['confidence_score']:.3f})")
            
            return result
            
        except Exception as e:
            print(f"  Error calling Gemini API: {e}")
            return {
                'contradiction_detected': False,
                'label': 'error',
                'confidence_score': 0.0,
                'explanation': f"API Error: {str(e)}",
                'processing_time_seconds': time.time() - start_time,
                'model_used': self.model_name,
                'raw_response': None
            }
    
    def _create_prompt(self, sentence: str, narrative: str) -> str:
        """Generalized NLI prompt with categorical logic boundaries."""
        prompt = f"""
# ==========================================================
# TASK: FORMAL LOGICAL INFERENCE (NLI)
# ==========================================================
Analyze the relationship between the NARRATIVE and the STATEMENT.

# ----------------------------------------------------------
# CATEGORICAL LOGIC DEFINITIONS
# ----------------------------------------------------------
1. **CLOTHING CATEGORY**
   - **REVEALING**: Requires explicit mention of the attire being minimal, provocative, or intentionally exposing. 
   - **MODEST**: Requires explicit mention of the attire being full-coverage, professional, or conservative.
   - **RULE**: Descriptions of physical states (e.g., being in bed, swimming, or changing) do not inherently map to "Revealing" or "Modest" unless the narrator assigns that quality to the outfit. If the narrative describes clothing but not the *category*, result is NEUTRAL.

2. **INTOXICATION CATEGORY**
   - **INTOXICATED**: Explicit mention of alcohol/drug consumption or physical symptoms of being "high" or "drunk."
   - **SOBER**: Explicit mention of total abstinence or being the "non-drinker" in a situation.
   - **RULE**: Cognitive states (e.g., "confusion," "fear," "shock") are NOT evidence of intoxication. If substances are not mentioned, result is NEUTRAL.

3. **RESISTANCE CATEGORY**
   - **RESISTED**: Requires active verbal "No," physical struggle, or attempted flight.
   - **YIELDED**: Defined as the *absence of resistance*. This includes freezing, compliance due to fear, or being unable to act.
   - **RULE**: If the narrator says "I didn't know what to do" or "I just stood there," this ENTAILS "yielded" and CONTRADICTS "resisted."

# ----------------------------------------------------------
# DATA INPUTS
# ----------------------------------------------------------
**NARRATIVE:**
{narrative}

**STATEMENT:**
{sentence}

# ----------------------------------------------------------
# RESPONSE REQUIREMENTS
# ----------------------------------------------------------
Classify the statement as:
- **ENTAILMENT**: Statement is a direct consequence of the narrative facts.
- **CONTRADICTION**: Statement is logically impossible given the narrative facts.
- **NEUTRAL**: The narrative is silent or lacks the specific categorical descriptors.

ANSWER: [CONTRADICTION / NEUTRAL / ENTAILMENT]
CONFIDENCE: [0.0 to 1.0]
EXPLANATION: [Identify the explicit fact or state the specific absence of information.]
"""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> Dict[str, any]:
        """Parse Gemini's response into structured format."""
        
        # Extract ANSWER
        answer_match = re.search(r'ANSWER:\s*(CONTRADICTION|NEUTRAL|ENTAILMENT)', 
                                response_text, re.IGNORECASE)
        if answer_match:
            label = answer_match.group(1).upper()
        else:
            # Fallback: check for keywords
            response_lower = response_text.lower()
            if 'contradiction' in response_lower:
                label = 'CONTRADICTION'
            elif 'entailment' in response_lower or 'support' in response_lower:
                label = 'ENTAILMENT'
            else:
                label = 'NEUTRAL'
        
        # Extract CONFIDENCE
        confidence_match = re.search(r'CONFIDENCE:\s*(0\.\d+|1\.0|1)', response_text)
        if confidence_match:
            confidence = float(confidence_match.group(1))
        else:
            # Default confidence based on clarity of language
            if any(word in response_text.lower() for word in ['clearly', 'definitely', 'explicitly']):
                confidence = 0.8
            elif any(word in response_text.lower() for word in ['possibly', 'might', 'somewhat']):
                confidence = 0.5
            else:
                confidence = 0.6
        
        # Extract EXPLANATION
        explanation_match = re.search(r'EXPLANATION:\s*(.+?)(?:\n|$)', 
                                     response_text, re.DOTALL)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        else:
            explanation = response_text
        
        return {
            'contradiction_detected': (label == 'CONTRADICTION'),
            'label': label.lower(),
            'confidence_score': confidence,
            'explanation': explanation
        }
    
    def print_results(self, results: Dict[str, any]):
        """Pretty print the results."""
        print("\n" + "="*80)
        print("GEMINI CONTRADICTION DETECTION RESULTS")
        print("="*80)
        
        print(f"\nContradiction Detected: {results['contradiction_detected']}")
        print(f"Label: {results['label']}")
        print(f"Confidence Score: {results['confidence_score']:.3f}")
        
        if results.get('explanation'):
            print(f"\nExplanation:")
            print(f"  {results['explanation']}")
        
        print(f"\nProcessing time: {results['processing_time_seconds']:.2f}s")
        print("="*80)


def process_narratives_with_gemini(df, narrative_column, detector, 
                                   use_sample=True, sample_size=100):
    """
    Process narratives against myth variations using Gemini.
    Same interface as the NLI version.
    
    Args:
        df: DataFrame with narratives
        narrative_column: Name of column containing narratives
        detector: GeminiContradictionDetector instance
        use_sample: Whether to use stratified sample
        sample_size: Number of narratives to sample
    
    Returns:
        DataFrame with results
    """
    from ContradictionDetectorNLI import create_stratified_sample
    
    # Import myth variations
    # from process_myths import (myth_variations_d1, myth_variations_d2)
    
    # Stratified sampling if requested
    if use_sample and len(df) > sample_size:
        print(f"\n{'='*80}")
        print(f"CREATING STRATIFIED SAMPLE OF {sample_size} NARRATIVES")
        print(f"{'='*80}")
        df_to_process = create_stratified_sample(
            df,
            text_column=narrative_column,
            n_samples=sample_size,
            n_bins=10,
            random_state=42
        )
    else:
        print(f"\nProcessing full dataset: {len(df)} narratives")
        df_to_process = df
    
    # Process each narrative
    print(f"\n{'='*80}")
    print("PROCESSING NARRATIVES WITH GEMINI")
    print(f"{'='*80}")
    print(f"Total narratives: {len(df_to_process)}")
    print(f"Sentences per narrative: 32 (4 types × 4 variations × 1 dose)")
    print(f"Total API calls: {len(df_to_process) * 32}")
    print(f"{'='*80}\n")
    
    results_list = []
    total_start_time = time.time()
    total_sentences_processed = 0
    api_errors = 0
    
    for idx, row in df_to_process.iterrows():
        narrative_start = time.time()
        
        print(f"\n{'='*80}")
        print(f"NARRATIVE {idx + 1}/{len(df_to_process)} (Original index: {idx})")
        print(f"{'='*80}")
        
        narrative = str(row[narrative_column])
        narrative_length = len(narrative)
        
        print(f"Narrative length: {narrative_length:,} characters")
        
        # Process each myth variation
        for dose_idx, myth_variations in enumerate([myth_variations_d1], start=1):
        # for dose_idx, myth_variations in enumerate([myth_variations_d1, myth_variations_d2], start=1):
            for myth_var, myth_details in myth_variations.items():
                for myth_type, myth_sentence in myth_details.items():
                    
                    sentence_start = time.time()
                    total_sentences_processed += 1
                    
                    print(f"\n  [{dose_idx}/2] {myth_var} - {myth_type}")
                    print(f"  Sentence: {myth_sentence[:80]}{'...' if len(myth_sentence) > 80 else ''}")
                    
                    # Run contradiction detection with Gemini
                    result = detector.detect_contradiction(
                        myth_sentence,
                        narrative,
                        return_explanation=True,
                        verbose=True
                    )
                    
                    # Track errors
                    if result['label'] == 'error':
                        api_errors += 1
                    
                    # Extract results
                    row_result = {
                        'narrative_index': idx,
                        'narrative_length': narrative_length,
                        'narrative': narrative,
                        'myth_type': myth_type,
                        'myth_variation': myth_var,
                        'dose': dose_idx,
                        'sentence': myth_sentence,
                        'contradiction_detected': result['contradiction_detected'],
                        'overall_label': result['label'],
                        'confidence_score': result['confidence_score'],
                        'explanation': result.get('explanation', ''),
                        'processing_time_seconds': result['processing_time_seconds'],
                        'model_used': result['model_used']
                    }
                    
                    results_list.append(row_result)
                    
                    # Progress info
                    sentence_time = time.time() - sentence_start
                    print(f"  → {result['label']} (conf: {result['confidence_score']:.3f}, "
                          f"time: {sentence_time:.1f}s)")
                    
                    # Rate limiting (Gemini has quota limits)
                    time.sleep(0.5)  # 2 requests per second max
        
        # Narrative summary
        narrative_time = time.time() - narrative_start
        elapsed_total = time.time() - total_start_time
        avg_time_per_narrative = elapsed_total / (idx - df_to_process.index[0] + 1)
        remaining_narratives = len(df_to_process) - (idx - df_to_process.index[0] + 1)
        eta_minutes = (remaining_narratives * avg_time_per_narrative) / 60
        
        print(f"\n  ✓ Narrative completed in {narrative_time:.1f}s")
        print(f"  ⏱️  Total elapsed: {elapsed_total/60:.1f}min | ETA: {eta_minutes:.1f}min")
        print(f"  ⚠️  API errors so far: {api_errors}")
    
    # Create results dataframe
    results_df = pd.DataFrame(results_list)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f'gemini_nli_results_{timestamp}.csv'
    results_df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"Total processing time: {(time.time() - total_start_time)/60:.2f} minutes")
    print(f"Total sentences processed: {total_sentences_processed}")
    print(f"API errors: {api_errors}")
    
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total rows: {len(results_df)}")
    print(f"Unique narratives: {results_df['narrative_index'].nunique()}")
    
    # Filter out errors for statistics
    valid_results = results_df[results_df['overall_label'] != 'error']
    
    print(f"\nValid results: {len(valid_results)} ({len(valid_results)/len(results_df)*100:.1f}%)")
    print(f"Contradictions detected: {valid_results['contradiction_detected'].sum()} "
          f"({valid_results['contradiction_detected'].sum()/len(valid_results)*100:.1f}%)")
    
    print(f"\nLabel distribution:")
    for label, count in valid_results['overall_label'].value_counts().items():
        print(f"  {label}: {count} ({count/len(valid_results)*100:.1f}%)")
    
    print(f"\nConfidence scores:")
    print(f"  Mean: {valid_results['confidence_score'].mean():.3f}")
    print(f"  Median: {valid_results['confidence_score'].median():.3f}")
    print(f"  Std: {valid_results['confidence_score'].std():.3f}")
    
    print(f"\nBy myth type:")
    for myth_type in valid_results['myth_type'].unique():
        subset = valid_results[valid_results['myth_type'] == myth_type]
        contra_rate = subset['contradiction_detected'].sum() / len(subset) * 100
        print(f"  {myth_type}: {contra_rate:.1f}% contradictions")
    
    print(f"\nBy variation:")
    for variation in valid_results['myth_variation'].unique():
        subset = valid_results[valid_results['myth_variation'] == variation]
        contra_rate = subset['contradiction_detected'].sum() / len(subset) * 100
        print(f"  {variation}: {contra_rate:.1f}% contradictions")
    
    return results_df

NEG_MYTH_d1 = {
    "clothing": "I was not wearing a revealing outfit.",
    "victim_intoxication": "I was not intoxicated.",
    "perpetrator_intoxication": "The perpetrator was not intoxicated.",
    "resistance": "I resisted the perpetrator.",
}

NEG_NON_MYTH_d1 = {
    "clothing": "I was not wearing a modest outfit.",
    "victim_intoxication": "I was not sober.",
    "perpetrator_intoxication": "The perpetrator was not sober.",
    "resistance": "I yielded to the perpetrator."
}

POS_MYTH_d1 = {
    "clothing": "I was wearing a revealing outfit.",
    "victim_intoxication": "I was intoxicated.",
    "perpetrator_intoxication": "The perpetrator was intoxicated.",
    "resistance": "I did not resist the perpetrator.",
}

POS_NON_MYTH_d1 = {
    "clothing": "I was wearing a modest outfit.",
    "victim_intoxication": "I was sober.",
    "perpetrator_intoxication": "The perpetrator was sober.",
    "resistance": "I did not yield to the perpetrator.",
}

myth_variations_d1 = {
    "NegMyth": NEG_MYTH_d1,
    "NegNonMyth": NEG_NON_MYTH_d1,
    "PosMyth": POS_MYTH_d1,
    "PosNonMyth": POS_NON_MYTH_d1
}

# Initialize Gemini detector
print("Initializing Gemini Contradiction Detector...")

# Set API key
api_key = "AIzaSyDE4rYnXZgvnqqwt7tRLs8Yo-Za6u4mGtE"
if not api_key:
    raise ValueError("Please set GEMINI_API_KEY environment variable or hardcode it")

detector = GeminiContradictionDetector(
    api_key=api_key,
    model_name="gemini-2.5-flash"
)

# Load your dataset
print("\nLoading dataset...")
df = pd.read_csv('../A_WebConf2025/8_Projections/RedditNarratives/Reddit_Dataset.csv')
df['Content'] = "TITLE: " + df['Title'] + "\n\nNARRATIVE BODY: " + df['Text']
narrative_column = 'Content'  

# Process with stratified sampling (100 narratives for testing)
results = process_narratives_with_gemini(
    df=df,
    narrative_column=narrative_column,
    detector=detector,
    use_sample=True,
    sample_size=360
)

print("\n✓ Processing complete!")
