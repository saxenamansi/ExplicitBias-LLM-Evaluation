#!/bin/bash
#SBATCH --job-name=AdviceGeneration_%A_%a
#SBATCH --output=log/AdviceGeneration_%A_%a.out
#SBATCH --error=log/AdviceGeneration_%A_%a.err
#SBATCH --array=0-4

conda init
source ~/.bashrc
conda activate summary_env

# ── Model selection via array index ───────────────────────────────────────────
MODELS=(gemma llama phi qwen mistral)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "========================================"
echo "Model: $MODEL  (array task $SLURM_ARRAY_TASK_ID)"
echo "========================================"

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR=/mnt/beegfs/msaxena4/6_Explicit-Implicit-Bias/4_Evaluation
NLI="/mnt/beegfs/msaxena4/6_Explicit-Implicit-Bias/Data/SentenceNLI-(SV)-Sample360.csv"

ADV_RESULTS=/mnt/beegfs/msaxena4/6_Explicit-Implicit-Bias/Results/AdviceGeneration/SampleResults_3PromptVariants

BASE=/mnt/beegfs/msaxena4/6_Explicit-Implicit-Bias/ResultAnalysis/AdviceGeneration/SampleResults
CONTEXT_APPEND_EMB=${BASE}/ContextAppend/${MODEL}_Embeddings.pkl
ORIGINAL_EMB=${BASE}/Original/${MODEL}_Embeddings.pkl

SHIFT_DIR=${BASE}/1_SemanticShift
PROJ_DIR=${BASE}/2_Projection

mkdir -p log \
    ${BASE}/ContextAppend \
    ${BASE}/Original \
    ${SHIFT_DIR} \
    ${PROJ_DIR}

# ── 1a: ContextAppend embeddings ──────────────────────────────────────────────
echo "--- 1a: ContextAppend embeddings ---"
python ${SCRIPT_DIR}/1a_ContextAppend-GetResponses.py \
    --t1    ${ADV_RESULTS}/${MODEL}_advice_t1.csv \
    --t2    ${ADV_RESULTS}/${MODEL}_advice_t2.csv \
    --nli   "${NLI}" \
    --model ${MODEL} \
    --task  AdviceGeneration
if [ $? -ne 0 ]; then echo "ERROR: 1a failed for $MODEL"; exit 1; fi

# ── 1b: Original embeddings ───────────────────────────────────────────────────
echo "--- 1b: Original embeddings ---"
python ${SCRIPT_DIR}/1b_Original-NoContext-GetResponses.py \
    --t1    ${ADV_RESULTS}/${MODEL}_advice_t1.csv \
    --nli   "${NLI}" \
    --model ${MODEL} \
    --task  AdviceGeneration
if [ $? -ne 0 ]; then echo "ERROR: 1b failed for $MODEL"; exit 1; fi