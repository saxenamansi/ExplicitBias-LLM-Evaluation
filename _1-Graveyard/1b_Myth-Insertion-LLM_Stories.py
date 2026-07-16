import os, sys
import pandas as pd
import numpy as np
import random
import torch
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM

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


def insert_myths(narrative_df, col):
    # Store combined results
    combined_results = []
    
    # Injection of myth sentences
    for i in range(NUM_NARRATIVES):
        
        narrative = narrative_df[col].iloc[i]
        sentences = sent_tokenize(narrative)
        
        for sent_idx in range(len(sentences)+1):
            story_outline = narrative_df['StoryOutline'].iloc[i]
            
            for j, myth_variations in enumerate([myth_variations_d1, myth_variations_d2]):
                for myth_var, myth_details in myth_variations.items():
                    for myth_type, myth_detail in myth_details.items():
                        modified_sentences = sentences[:sent_idx] + [myth_detail] + sentences[sent_idx:]
                        modified_narrative = " ".join(sentences[:sent_idx] + [myth_detail] + sentences[sent_idx:])
                        row = {
                            "narrative_idx" : i,
                            "StoryOutline" : story_outline,
                            "original_narrative": narrative,
                            "myth_type": myth_type,
                            "myth_variation" : myth_var,
                            "dose" : j+1,
                            "myth_detail": myth_detail,
                            "index": sent_idx,
                            "modified_narrative": modified_narrative, 
                            "modified_sentences": modified_sentences,
                            "myth_sentence_idx": sent_idx,
                            "prev_sentence_idx": sent_idx - 1 if sent_idx > 0 else None,
                            "next_sentence_idx": sent_idx + 1 if sent_idx < len(modified_sentences) - 1 else None,
                        }
                        combined_results.append(row)
    df_out = pd.DataFrame(combined_results)
    return df_out

gemini_df = pd.read_csv('../2_GeneratingNarratives/Results/Gemini_processed.csv')
llama_df = pd.read_csv('../2_GeneratingNarratives/Results/Llama_processed.csv')
mistral_df = pd.read_csv('../2_GeneratingNarratives/Results/Mistral-100.csv')
NUM_NARRATIVES = 100

gemini_out = insert_myths(gemini_df, 'OriginalNarrative')
llama_out = insert_myths(llama_df, 'ProcessedNarrative')
mistral_out = insert_myths(mistral_df, 'OriginalNarrative')

gemini_out.to_csv("Results/Gemini_candidates.csv", index=False)
llama_out.to_csv("Results/Llama_candidates.csv", index=False)
mistral_out.to_csv("Results/Mistral_candidates.csv", index=False)