#!/bin/bash
# Capped LoRA DPO + Echo SFT: 10 concepts × 50 trials = 500 trials/model.
# Supports --append to resume after an early stop.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$PWD/hf-cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_STEM="${RUN_STEM:-runpod-lora-40c-50t}"
CKPT_DIR="${CKPT_DIR:-results/ckpts/runpod-olmo32-dolci-paper}"
EVAL_DIR="results/eval/${RUN_STEM}"
LOG="results/logs/${RUN_STEM}-500cap.log"
BASE_MODEL="${BASE_MODEL:-allenai/Olmo-3.1-32B-Instruct-SFT}"
NUM_CONCEPTS="${NUM_CONCEPTS:-10}"
TRIALS_PER_CONCEPT="${TRIALS_PER_CONCEPT:-50}"
MAX_TRIALS="${MAX_TRIALS:-500}"
EVAL_LAYER="${EVAL_LAYER:-25}"
EVAL_ALPHA="${EVAL_ALPHA:-4.0}"
PHASE="${PHASE:-both}"  # dpo | echo | both

mkdir -p results/logs "$EVAL_DIR" "$HF_HOME"

COMMON_ARGS=(
  --output-dir "$EVAL_DIR"
  --layer "$EVAL_LAYER"
  --alpha "$EVAL_ALPHA"
  --num-concepts "$NUM_CONCEPTS"
  --trials-per-concept "$TRIALS_PER_CONCEPT"
  --max-trials-per-model "$MAX_TRIALS"
  --max-length 2048
  --max-new-tokens 100
  --temperature 1.0
  --top-k 50
  --top-p 1.0
  --use-cache
  --append
)

DPO_MODEL="LoRA_DPO=${BASE_MODEL}:${CKPT_DIR}/dpo_adapter"
ECHO_MODEL="Echo_SFT=${BASE_MODEL}:${CKPT_DIR}/echo_sft_adapter"

exec >> "$LOG" 2>&1
echo "=== 500-trial capped eval $(date) PHASE=$PHASE RUN_STEM=$RUN_STEM ==="

if [[ "$PHASE" == "dpo" || "$PHASE" == "both" ]]; then
  python -u eval_injection.py --models "$DPO_MODEL" "${COMMON_ARGS[@]}"
fi

if [[ "$PHASE" == "echo" || "$PHASE" == "both" ]]; then
  python -u eval_injection.py --models "$ECHO_MODEL" "${COMMON_ARGS[@]}"
fi

python -u grade_eval.py \
  --results "$EVAL_DIR/eval_results.jsonl" \
  --output-dir "$EVAL_DIR" \
  --heuristic-only

echo "=== 500-trial capped eval done $(date) ==="
cat "$EVAL_DIR/grading_summary.json"
