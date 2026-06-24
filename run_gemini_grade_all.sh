#!/bin/bash
# Gemini 3.1 Flash Lite grading for LoRA DPO, Echo SFT, and KTO (paper judge prompts).
# Batches 10 trials per API request (~10x fewer calls vs per-trial grading).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-.venv-grading/bin/python}"
API_KEY_FILE="${API_KEY_FILE:-geminikey_txt.txt}"
MODEL="${GEMINI_MODEL:-gemini-3.1-flash-lite}"
BATCH_SIZE="${GEMINI_BATCH_SIZE:-10}"
LOG="results/logs/gemini-grading-all.log"
mkdir -p results/logs

exec > >(tee -a "$LOG") 2>&1
echo "=== Gemini grading all models start $(date) model=$MODEL ==="

grade_one() {
  local results="$1"
  local model="$2"
  echo "--- $model from $results ---"
  "$PY" grade_eval.py --gemini \
    --judge-model "$MODEL" \
    --api-key-file "$API_KEY_FILE" \
    --gemini-batch-size "$BATCH_SIZE" \
    --results "$results" \
    --models "$model"
}

# DPO + Echo share runpod-lora-40c-50t (DPO/KTO: 25 concepts; Echo: 10 concepts only).
grade_one results/eval/runpod-lora-40c-50t/eval_results.jsonl LoRA_DPO
grade_one results/eval/runpod-lora-40c-50t/eval_results.jsonl Echo_SFT
grade_one results/eval/runpod-kto-negative-500t/eval_results.jsonl KTO_Negative

echo "=== Gemini grading all models done $(date) ==="
