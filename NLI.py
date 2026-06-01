"""
Feature detection via NLI (sentence-level detection).
We detect presence of myths, narrative perspective, age, gender, perpetrator's relationship to victim. 
"""

import os, sys
import pandas as pd
import numpy as np

from SentenceContradictionDetectorNLI import SentenceLevelContradictionDetector, create_stratified_sample

from huggingface_hub import login
login(token="xyz") # HuggingFace Token

MYTH = {
    "clothing": "The victim was wearing revealing or sexually provocative clothing that exposed their body.",
    "victim_intoxication": "The victim was intoxicated due to usage of alcohol or other drugs of abuse.",
    "perpetrator_intoxication": "The perpetrator was intoxicated due to usage of alcohol or other drugs of abuse.",
    "resistance": "The victim did not physically or verbally resist the perpetrator."
}

PERSPECTIVE = {
    "first_person_victim": "The author is a sexual violence victim.",
    "third_person_victim": "The author is talking about a sexual violence victim, not themselves.",
    "first_person_perpetrator":"The author is a sexual violence perpetrator.",
    "third_person_perpetrator": "The author is talking about a sexual violence perpetrator, not themselves."  
}

GENDER = {
    "victim_female": "The victim of the sexual violence is a woman or female.",
    "victim_male": "The victim of the sexual violence is a man or male.",
    "perpetrator_female": "The perpetrator of the sexual violence is a woman or female.",
    "perpetrator_male": "The perpetrator of the sexual violence is a man or male."
}

AGE = {
    "childhood_abuse": "The sexual violence occurred when the victim was a child or minor under 18.",
    "adult_victim": "The sexual violence occurred when the victim was an adult over 18.",
}

RELATIONSHIP = {
    "stranger_assault": "The sexual violence perpetrator was a stranger the victim did not know prior to the incident.",
    "acquaintance_assault": "The sexual violence perpetrator was an acquaintance, friend, coworker, or someone the victim knew casually before the incident.",
    "intimate_partner": "The sexual violence perpetrator was the victim's current or former romantic partner, spouse, or boyfriend/girlfriend.",
    "family_member": "The sexual violence perpetrator was a family member of the victim."
}


def sample_narratives_for_validation(results_df, n_narratives=100, random_state=42):
    # Get unique narrative indices
    unique_narratives = results_df['narrative_index'].unique()
    print(f"Total narratives available: {len(unique_narratives)}")
    
    # Sample N narrative indices randomly
    sampled_indices = pd.Series(unique_narratives).sample(
        n=min(n_narratives, len(unique_narratives)), 
        random_state=random_state
    ).values
    
    # Get all rows for those narratives
    sampled_df = results_df[results_df['narrative_index'].isin(sampled_indices)].copy()    
    return sampled_df

def process_narratives_sentence_level(df, narrative_column, detector, sample_size=100):
    """
    Process narratives with sentence-level contradiction detection.
    
    Args:
        df: DataFrame with narratives
        narrative_column: Name of column containing narratives
        detector: SentenceLevelContradictionDetector instance
        sample_size: Number of narratives to sample
    
    Returns:
        DataFrame with results
    """
    
    df_to_process = df.copy()
    
    # Process each narrative
    print(f"\n{'='*80}")
    print("SENTENCE-LEVEL CONTRADICTION DETECTION")
    print(f"{'='*80}")
    print(f"Total narratives: {len(df_to_process)}")
    print(f"Myth categories: 2 (MYTH, ANTI-MYTH)")
    print(f"Myth types: 4 (clothing, victim_intox, perp_intox, resistance)")
    print(f"Total tests per narrative: 8 (4 types × 2 categories)")
    print(f"Total inference operations: {len(df_to_process) * 8}")
    print(f"{'='*80}\n")
    
    results_list = []
    total_tests_processed = 0
    
    for idx, row in df_to_process.iterrows():
        
        print(f"\n{'='*80}")
        print(f"NARRATIVE {idx + 1}/{len(df_to_process)} (Original index: {idx})")
        print(f"{'='*80}")
        
        narrative = str(row[narrative_column])
        narrative_length = len(narrative)
        
        print(f"Narrative length: {narrative_length:,} characters")
        
        # Test all categories
        for category, sentences_dict in [("MYTH", MYTH), ("PERSPECTIVE", PERSPECTIVE), ("AGE", AGE), ("GENDER", GENDER), ("RELATIONSHIP", RELATIONSHIP)]:
            
            print(f"\n  Testing {category} sentences:")
            
            for myth_type, myth_sentence in sentences_dict.items():
                
                total_tests_processed += 1
                
                print(f"\n    [{myth_type}]")
                print(f"    Sentence: {myth_sentence[:80]}{'...' if len(myth_sentence) > 80 else ''}")
                
                # Run sentence-level detection
                result = detector.detect_contradiction(
                    myth_sentence,
                    narrative,
                    return_all_sentences=True,
                    verbose=True
                )
                
                # Extract results
                row_result = {
                    'narrative_index': idx,
                    'narrative_length': narrative_length,
                    'narrative': narrative,
                    'myth_type': myth_type,
                    'myth_category': category,
                    'sentence': myth_sentence,
                    'contradiction_detected': result['contradiction_detected'],
                    'entailment_detected': result.get('entailment_detected', False), 
                    'overall_label': result['label'],
                    'confidence_score': result['confidence_score'],
                    'entailment_confidence_score': result.get('entailment_confidence_score', 0.0),  
                    'total_sentences': result['total_sentences'],
                    'contradicting_sentences_count': result['contradicting_sentences_count'],
                    'entailing_sentences_count': result.get('entailing_sentences_count', 0),  
                }
                
                # Contradicting sentence
                if result['contradicting_sentences']:
                    # Get top contradicting sentence
                    top_contra = result['contradicting_sentences'][0]
                    row_result['top_contradicting_sentence'] = top_contra['sentence']
                    row_result['top_contradicting_score'] = top_contra['contradiction_score']
                    
                    # Store ALL contradicting sentences (as string for CSV)
                    all_contradicting = [
                        f"[Score: {c['contradiction_score']:.3f}] {c['sentence']}"
                        for c in result['contradicting_sentences']
                    ]
                    row_result['all_contradicting_sentences'] = " ||| ".join(all_contradicting)
                    row_result['num_contradicting_sentences'] = len(result['contradicting_sentences'])
                else:
                    row_result['top_contradicting_sentence'] = None
                    row_result['top_contradicting_score'] = None
                    row_result['all_contradicting_sentences'] = None
                    row_result['num_contradicting_sentences'] = 0
                
                # Entailing sentence
                if result.get('entailing_sentences'):
                    # Get top entailing sentence
                    top_entail = result['entailing_sentences'][0]
                    row_result['top_entailing_sentence'] = top_entail['sentence']
                    row_result['top_entailing_score'] = top_entail['entailment_score']
                    
                    # Store ALL entailing sentences
                    all_entailing = [
                        f"[Score: {e['entailment_score']:.3f}] {e['sentence']}"
                        for e in result['entailing_sentences']
                    ]
                    row_result['all_entailing_sentences'] = " ||| ".join(all_entailing)
                    row_result['num_entailing_sentences'] = len(result['entailing_sentences'])
                else:
                    row_result['top_entailing_sentence'] = None
                    row_result['top_entailing_score'] = None
                    row_result['all_entailing_sentences'] = None
                    row_result['num_entailing_sentences'] = 0
                
                # Probability scores
                # NOTE: We take MAX scores across all sentences, NOT average
                # This tells us: "Did ANY sentence strongly show this label?"
                # - max_contradiction_prob: highest contradiction score from any sentence
                # - max_neutral_prob: highest neutral score from any sentence  
                # - max_entailment_prob: highest entailment score from any sentence
                # These are independent - different sentences can have different max scores
                if result.get('all_sentence_results'):
                    # Get max scores across all sentences
                    all_results = result['all_sentence_results']
                    
                    max_contradiction = max([r['result']['contradiction_score'] for r in all_results])
                    max_neutral = max([r['result']['scores'].get('NEUTRAL', r['result']['scores'].get('neutral', 0.0)) 
                                      for r in all_results])
                    max_entailment = max([r['result']['scores'].get('ENTAILMENT', r['result']['scores'].get('entailment', 0.0)) 
                                         for r in all_results])
                    
                    row_result['max_contradiction_prob'] = max_contradiction
                    row_result['max_neutral_prob'] = max_neutral
                    row_result['max_entailment_prob'] = max_entailment
                
                results_list.append(row_result)
            
    # Create results dataframe
    results_df = pd.DataFrame(results_list)
    
    # Print summary
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total tests processed: {total_tests_processed}")
    
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
    
    print(f"\nBy category:")
    for category in results_df['myth_category'].unique():
        subset = results_df[results_df['myth_category'] == category]
        contra_rate = subset['contradiction_detected'].sum() / len(subset) * 100
        print(f"  {category}: {contra_rate:.1f}% contradictions")
    
    return df, results_df

print("Initializing Sentence-Level Contradiction Detector...")

# NLI model
detector = SentenceLevelContradictionDetector(
    model_name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    contradiction_threshold=0.5
)

# Load dataset
print("\nLoading dataset...")
df = pd.read_csv('Reddit-SV-Data.csv') 
df['Content'] = df['Title'] + " [SEP] " + df['Text']
narrative_column = 'Content'

# Process full dataset
full_df, full_results = process_narratives_sentence_level(
    df=df,
    narrative_column=narrative_column,
    detector=detector,
    use_sample=False,
)
full_df.to_csv('Results/Reddit-SV-Full.csv', index=False)
full_results.to_csv('Results/SentenceNLI-SV-Full.csv', index=False)

print("\n✓ Processing complete!")
