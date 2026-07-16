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
NLI-based Contradiction Detector for Reddit Posts
Detects if myth sentence contradicts any part of long Reddit narratives using semantic chunking.
"""

import pandas as pd
import time
from ContradictionDetectorNLI import ContradictionDetector, create_stratified_sample

# Define myth variations
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

NEG_MYTH_d2 = {
    "clothing": "I was not wearing a revealing outfit. My outfit did not expose much of my skin and body shape. Others observed that my clothing was not bold in style and did not have minimum coverage.",
    "victim_intoxication": "I was not intoxicated. I had not indulged in consuming alcohol, and my senses were not impaired. Others observed that I did not have an unsteady walk, slurred speech, and blurred vision.",
    "perpetrator_intoxication": "The perpetrator was not intoxicated. The perpetrator had not indulged in consuming alcohol, and their senses were not impaired. Others observed that the perpetrator did not have an unsteady walk, slurred speech, and blurred vision.",
    "resistance": "I resisted the perpetrator. I pushed them and stepped away to prevent them from taking control. I shouted for the perpetrator to stop and hindered them from continuing."
}

NEG_NON_MYTH_d2 = {
    "clothing": "I was not wearing a modest outfit. My outfit did not cover much of my skin and body shape. Others observed that my clothing was not conservative in style and did not have maximum coverage.",
    "victim_intoxication": "I was not sober. I had not refrained from consuming alcohol, and my senses were not unaffected. Others observed that I did not have a stable walk, articulate speech, and clear vision.",
    "perpetrator_intoxication": "The perpetrator was not sober. The perpetrator had not refrained from consuming alcohol, and their senses were not unaffected. Others observed that the perpetrator did not have a stable walk, articulate speech, and clear vision.",
    "resistance": "I yielded to the perpetrator. I allowed them to take control and stood still. I remained silent and let the perpetrator continue unhindered."
}

POS_MYTH_d2 = {
    "clothing": "I was wearing a revealing outfit. My outfit exposed much of my skin and body shape. Others observed that my clothing was bold in style and had minimum coverage.",
    "victim_intoxication": "I was intoxicated. I had indulged in consuming alcohol, and my senses were impaired. Others observed that I had an unsteady walk, slurred speech, and blurred vision.",
    "perpetrator_intoxication": "The perpetrator was intoxicated. The perpetrator had indulged in consuming alcohol, and their senses were impaired. Others observed that the perpetrator had an unsteady walk, slurred speech, and blurred vision.",
    "resistance": "I did not resist the perpetrator. I did not push them and stepped away to prevent them from taking control. I did not shout for the perpetrator to stop and did not hinder them from continuing."
}

POS_NON_MYTH_d2 = {
    "clothing": "I was wearing a modest outfit. My outfit covered much of my skin and body shape. Others observed that my clothing was conservative in style and had maximum coverage.",
    "victim_intoxication": "I was sober. I had refrained from consuming alcohol, and my senses were unaffected. Others observed that I had a stable walk, articulate speech, and clear vision.",
    "perpetrator_intoxication": "The perpetrator was sober. The perpetrator had refrained from consuming alcohol, and their senses were unaffected. Others observed that the perpetrator had a stable walk, articulate speech, and clear vision.",
    "resistance": "I did not yield to the perpetrator. I did not allow them to take control and did not stay still. I did not remain silent and did not let the perpetrator continue unhindered."
}

myth_variations_d2 = {
    "NegMyth": NEG_MYTH_d2,
    "NegNonMyth": NEG_NON_MYTH_d2,
    "PosMyth": POS_MYTH_d2,
    "PosNonMyth": POS_NON_MYTH_d2
}


def process_narratives_with_myths(df, narrative_column, detector, use_sample=True, sample_size=100):
    """
    Process narratives against all myth variation sentences.
    
    Args:
        df: DataFrame with narratives
        narrative_column: Name of column containing narratives
        detector: ContradictionDetector instance
        use_sample: Whether to use stratified sample
        sample_size: Number of narratives to sample
    
    Returns:
        DataFrame with results
    """
    
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
    print("PROCESSING NARRATIVES AGAINST MYTH VARIATIONS")
    print(f"{'='*80}")
    print(f"Total narratives: {len(df_to_process)}")
    print(f"Sentences per narrative: 32 (4 types × 4 variations × 2 doses)")
    print(f"Total inference operations: {len(df_to_process) * 32}")
    print(f"{'='*80}\n")
    
    results_list = []
    total_start_time = time.time()
    total_sentences_processed = 0
    
    for idx, row in df_to_process.iterrows():
        narrative_start = time.time()
        
        print(f"\n{'='*80}")
        print(f"NARRATIVE {idx + 1}/{len(df_to_process)} (Original index: {idx})")
        print(f"{'='*80}")
        
        narrative = str(row[narrative_column])
        narrative_length = len(narrative)
        
        print(f"Narrative length: {narrative_length:,} characters")
        
        # Process each myth variation
        for dose_idx, myth_variations in enumerate([myth_variations_d1, myth_variations_d2], start=1):
            for myth_var, myth_details in myth_variations.items():
                for myth_type, myth_sentence in myth_details.items():
                    
                    sentence_start = time.time()
                    total_sentences_processed += 1
                    
                    print(f"\n  [{dose_idx}/2] {myth_var} - {myth_type}")
                    print(f"  Sentence: {myth_sentence[:80]}{'...' if len(myth_sentence) > 80 else ''}")
                    
                    # Run contradiction detection
                    result = detector.detect_contradiction(
                        myth_sentence,
                        narrative,
                        return_all_chunks=True,
                        verbose=False  # Suppress per-chunk progress
                    )
                    
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
                        'total_chunks': result['total_chunks'],
                        'contradicting_chunks_count': result['contradicting_chunks_count'],
                        'processing_time_seconds': result.get('processing_time_seconds', 0),
                    }
                    
                    # Add probabilities
                    if result.get('all_chunk_results'):
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
                        
                        # Average probabilities
                        all_contra_probs = [cr['result']['contradiction_score'] for cr in result['all_chunk_results']]
                        row_result['avg_contradiction_prob'] = sum(all_contra_probs) / len(all_contra_probs)
                        
                        # Contradicting segment
                        if result['contradicting_segments']:
                            row_result['top_contradicting_segment'] = result['contradicting_segments'][0]['full_text']
                            row_result['top_contradicting_score'] = result['contradicting_segments'][0]['score']
                        else:
                            row_result['top_contradicting_segment'] = None
                            row_result['top_contradicting_score'] = None
                    
                    results_list.append(row_result)
                    
                    # Progress info
                    sentence_time = time.time() - sentence_start
                    print(f"  → {result['label']} (conf: {result['confidence_score']:.3f}, "
                          f"chunks: {result['total_chunks']}, time: {sentence_time:.1f}s)")
        
        # Narrative summary
        narrative_time = time.time() - narrative_start
        elapsed_total = time.time() - total_start_time
        avg_time_per_narrative = elapsed_total / (idx - df_to_process.index[0] + 1)
        remaining_narratives = len(df_to_process) - (idx - df_to_process.index[0] + 1)
        eta_minutes = (remaining_narratives * avg_time_per_narrative) / 60
        
        print(f"\n  ✓ Narrative completed in {narrative_time:.1f}s")
        print(f"  ⏱️  Total elapsed: {elapsed_total/60:.1f}min | ETA: {eta_minutes:.1f}min")
    
    # Create results dataframe
    results_df = pd.DataFrame(results_list)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f'nli_results_{timestamp}.csv'
    results_df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"Total processing time: {(time.time() - total_start_time)/60:.2f} minutes")
    print(f"Total sentences processed: {total_sentences_processed}")
    
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total rows: {len(results_df)}")
    print(f"Unique narratives: {results_df['narrative_index'].nunique()}")
    print(f"Contradictions detected: {results_df['contradiction_detected'].sum()} "
          f"({results_df['contradiction_detected'].sum()/len(results_df)*100:.1f}%)")
    
    print(f"\nLabel distribution:")
    for label, count in results_df['overall_label'].value_counts().items():
        print(f"  {label}: {count} ({count/len(results_df)*100:.1f}%)")
    
    print(f"\nConfidence scores:")
    print(f"  Mean: {results_df['confidence_score'].mean():.3f}")
    print(f"  Median: {results_df['confidence_score'].median():.3f}")
    print(f"  Std: {results_df['confidence_score'].std():.3f}")
    
    print(f"\nBy myth type:")
    for myth_type in results_df['myth_type'].unique():
        subset = results_df[results_df['myth_type'] == myth_type]
        contra_rate = subset['contradiction_detected'].sum() / len(subset) * 100
        print(f"  {myth_type}: {contra_rate:.1f}% contradictions")
    
    print(f"\nBy variation:")
    for variation in results_df['myth_variation'].unique():
        subset = results_df[results_df['myth_variation'] == variation]
        contra_rate = subset['contradiction_detected'].sum() / len(subset) * 100
        print(f"  {variation}: {contra_rate:.1f}% contradictions")
    
    return results_df


if __name__ == "__main__":
    # Initialize detector
    print("Initializing ContradictionDetector...")
    detector = ContradictionDetector(
        model_name="facebook/bart-large-mnli",
        max_chunk_tokens=400,
        contradiction_threshold=0.3
    )
    
    # Load dataset
    print("\nLoading dataset...")
    df = pd.read_csv('../A_WebConf2025/8_Projections/RedditNarratives/Reddit_Dataset.csv') 
    df['Content'] = df['Title'] + df['Text']
    # narrative_column = 'Content'  # or whatever your narrative column is called
    
    narrative_column = 'Content'
    
    # Process with stratified sampling
    results = process_narratives_with_myths(
        df=df,
        narrative_column=narrative_column,
        detector=detector,
        use_sample=True,
        sample_size=360
    )
    
    print("\n✓ Processing complete!")