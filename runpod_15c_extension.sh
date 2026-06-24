#!/bin/bash
# Append eval for the next 15 Lindsey concepts (11-25): 50 trials each (25 inj + 25 ctl).
# LoRA DPO -> runpod-lora-40c-50t; KTO negative -> runpod-kto-negative-500t.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$PWD/hf-cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BASE_MODEL="${BASE_MODEL:-allenai/Olmo-3.1-32B-Instruct-SFT}"
DPO_CKPT="${DPO_CKPT:-results/ckpts/runpod-olmo32-dolci-paper}"
KTO_CKPT="${KTO_CKPT:-results/ckpts/runpod-olmo32-kto-negative}"
DPO_EVAL_DIR="${DPO_EVAL_DIR:-results/eval/runpod-lora-40c-50t}"
KTO_EVAL_DIR="${KTO_EVAL_DIR:-results/eval/runpod-kto-negative-500t}"
LOG="results/logs/runpod-15c-extension.log"
TRIALS_PER_CONCEPT="${TRIALS_PER_CONCEPT:-50}"
EVAL_LAYER="${EVAL_LAYER:-25}"
EVAL_ALPHA="${EVAL_ALPHA:-4.0}"
PHASE="${PHASE:-both}"  # dpo | kto | both

# Lindsey concepts 11-25 (skip first 10 already evaluated).
EXTENSION_CONCEPTS=(
  Trees Avalanches Mirrors Fountains Quarries
  Sadness Xylophones Secrecy Oceans Happiness
  Deserts Kaleidoscopes Sugar Vegetables Poetry
)

mkdir -p results/logs "$DPO_EVAL_DIR" "$KTO_EVAL_DIR" "$HF_HOME"

COMMON_ARGS=(
  --concepts "${EXTENSION_CONCEPTS[@]}"
  --trials-per-concept "$TRIALS_PER_CONCEPT"
  --max-trials-per-model 0
  --layer "$EVAL_LAYER"
  --alpha "$EVAL_ALPHA"
  --max-length 2048
  --max-new-tokens 100
  --temperature 1.0
  --top-k 50
  --top-p 1.0
  --use-cache
  --append
)

DPO_MODEL="LoRA_DPO=${BASE_MODEL}:${DPO_CKPT}/dpo_adapter"
KTO_MODEL="KTO_Negative=${BASE_MODEL}:${KTO_CKPT}/kto_negative_adapter"

exec > >(tee -a "$LOG") 2>&1
echo "=== 15-concept Lindsey extension start $(date) PHASE=$PHASE ==="
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"

if [[ "$PHASE" == "dpo" || "$PHASE" == "both" ]]; then
  python -u eval_injection.py \
    --models "$DPO_MODEL" \
    --output-dir "$DPO_EVAL_DIR" \
    "${COMMON_ARGS[@]}"
  python -u grade_eval.py \
    --results "$DPO_EVAL_DIR/eval_results.jsonl" \
    --output-dir "$DPO_EVAL_DIR" \
    --heuristic-only
fi

if [[ "$PHASE" == "kto" || "$PHASE" == "both" ]]; then
  python -u eval_injection.py \
    --models "$KTO_MODEL" \
    --output-dir "$KTO_EVAL_DIR" \
    "${COMMON_ARGS[@]}"
  python -u grade_eval.py \
    --results "$KTO_EVAL_DIR/eval_results.jsonl" \
    --output-dir "$KTO_EVAL_DIR" \
    --heuristic-only
fi

echo "=== 15-concept Lindsey extension done $(date) ==="
echo "DPO eval lines: $(wc -l < "$DPO_EVAL_DIR/eval_results.jsonl")"
echo "KTO eval lines: $(wc -l < "$KTO_EVAL_DIR/eval_results.jsonl")"
if [[ -f "$DPO_EVAL_DIR/grading_summary.json" ]]; then
  echo "--- DPO grading ---"
  cat "$DPO_EVAL_DIR/grading_summary.json"
fi
if [[ -f "$KTO_EVAL_DIR/grading_summary.json" ]]; then
  echo "--- KTO grading ---"
  cat "$KTO_EVAL_DIR/grading_summary.json"
fi
