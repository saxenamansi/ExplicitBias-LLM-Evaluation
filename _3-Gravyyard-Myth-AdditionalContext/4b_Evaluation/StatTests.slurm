#!/bin/bash
#SBATCH --job-name=AdviceGeneration
#SBATCH --output=log/AdviceGeneration.out
#SBATCH --error=log/AdviceGeneration.err
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

BASE=/mnt/beegfs/msaxena4/6_Explicit-Implicit-Bias/ResultAnalysis/AdviceGeneration/SampleResults

CONTEXT_APPEND_EMB=${BASE}/ContextAppend/${MODEL}_WithEmbeddings.pkl
ORIGINAL_EMB=${BASE}/Original/${MODEL}_WithEmbeddings.pkl

SHIFT_DIR=${BASE}/1_SemanticShift
PROJ_DIR=${BASE}/2_Projection/

mkdir -p log ${SHIFT_DIR} ${PROJ_DIR}

# ── 1c: Semantic shift ────────────────────────────────────────────────────────
echo "--- 1c: Semantic shift ---"
python ${SCRIPT_DIR}/1c_SemanticShift.py \
    --ContextAppend ${CONTEXT_APPEND_EMB} \
    --Original      ${ORIGINAL_EMB} \
    --model         ${MODEL} \
    --task          AdviceGeneration
if [ $? -ne 0 ]; then echo "ERROR: 1c failed for $MODEL"; exit 1; fi

# ── 2b: Projection ────────────────────────────────────────────────────────────
echo "--- 2b: Projection ---"
python ${SCRIPT_DIR}/2b_Projection.py \
    --ContextAppend ${CONTEXT_APPEND_EMB} \
    --Original      ${ORIGINAL_EMB} \
    --model         ${MODEL} \
    --task          AdviceGeneration
if [ $? -ne 0 ]; then echo "ERROR: 2b failed for $MODEL"; exit 1; fi

# ── 2c: Myth alignment ────────────────────────────────────────────────────────
echo "--- 2c: Myth alignment ---"
python ${SCRIPT_DIR}/2c_MythAlignment.py \
    --ContextAppend ${PROJ_DIR}/${MODEL}_ContextAppend_Projections.pkl \

6_Explicit-Implicit-Bias/ResultAnalysis/AdviceGeneration/SampleResults/2_Projection/gemma_ContextAppend_Projections.pkl
6_Explicit-Implicit-Bias/ResultAnalysis/AdviceGeneration/SampleResults/2_Projection/gemma_Original_Projections.pkl

    --Original      ${PROJ_DIR}/${MODEL}_Original_Projections.pkl \
    --nli           "${NLI}" \
    --model         ${MODEL} \
    --task          AdviceGeneration
if [ $? -ne 0 ]; then echo "ERROR: 2c failed for $MODEL"; exit 1; fi

echo "--- $MODEL complete ---"