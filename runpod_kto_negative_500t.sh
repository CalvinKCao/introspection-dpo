#!/bin/bash
# Negative-only KTO LoRA on 5k rejected Dolci responses, then 500-trial concept-injection eval.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$PWD/hf-cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_STEM="${RUN_STEM:-runpod-kto-negative-500t}"
CKPT_DIR="${CKPT_DIR:-results/ckpts/runpod-olmo32-kto-negative}"
DATA_DIR="${DATA_DIR:-results/datasets/runpod-olmo32-kto-negative}"
EVAL_DIR="results/eval/${RUN_STEM}"
LOG="results/logs/${RUN_STEM}.log"
BASE_MODEL="${BASE_MODEL:-allenai/Olmo-3.1-32B-Instruct-SFT}"
NUM_SAMPLES="${NUM_SAMPLES:-5000}"
NUM_CONCEPTS="${NUM_CONCEPTS:-10}"
TRIALS_PER_CONCEPT="${TRIALS_PER_CONCEPT:-50}"
MAX_TRIALS="${MAX_TRIALS:-500}"
EVAL_LAYER="${EVAL_LAYER:-25}"
EVAL_ALPHA="${EVAL_ALPHA:-4.0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_GRADE="${RUN_GRADE:-1}"

mkdir -p results/logs "$CKPT_DIR" "$DATA_DIR" "$EVAL_DIR" "$HF_HOME"

if [[ "${SKIP_PIP_INSTALL:-0}" != "1" ]]; then
  pip install -q -r requirements-runpod.txt
fi

exec > >(tee -a "$LOG") 2>&1
echo "=== KTO negative-only experiment start $(date) RUN_STEM=$RUN_STEM BASE_MODEL=$BASE_MODEL ==="
python -c "import torch, trl, hf_olmo; print(torch.__version__, trl.__version__, torch.cuda.get_device_name(0))"

if [[ "$RUN_TRAIN" == "1" ]]; then
  python -u train_kto_negative.py \
    --base-model "$BASE_MODEL" \
    --dataset allenai/dolci-instruct-dpo \
    --dataset-split train \
    --output-dir "$CKPT_DIR" \
    --dataset-cache-dir "$DATA_DIR" \
    --num-samples "$NUM_SAMPLES" \
    --max-length 2048 \
    --max-prompt-length 1024 \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-dropout 0.0 \
    --target-modules q_proj,v_proj \
    --per-device-train-batch-size 2 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-5 \
    --beta 0.1 \
    --warmup-steps 50 \
    --lr-scheduler-type linear \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --num-train-epochs 1 \
    --save-steps 50 \
    --save-total-limit 5 \
    --save-every-minutes 60 \
    --resume
else
  echo "Skipping training because RUN_TRAIN=$RUN_TRAIN"
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  KTO_MODEL="KTO_Negative=${BASE_MODEL}:${CKPT_DIR}/kto_negative_adapter"
  python -u eval_injection.py \
    --models "$KTO_MODEL" \
    --output-dir "$EVAL_DIR" \
    --layer "$EVAL_LAYER" \
    --alpha "$EVAL_ALPHA" \
    --num-concepts "$NUM_CONCEPTS" \
    --trials-per-concept "$TRIALS_PER_CONCEPT" \
    --max-trials-per-model "$MAX_TRIALS" \
    --max-length 2048 \
    --max-new-tokens 100 \
    --temperature 1.0 \
    --top-k 50 \
    --top-p 1.0 \
    --use-cache \
    --append
else
  echo "Skipping eval because RUN_EVAL=$RUN_EVAL"
fi

if [[ "$RUN_GRADE" == "1" ]]; then
  python -u grade_eval.py \
    --results "$EVAL_DIR/eval_results.jsonl" \
    --output-dir "$EVAL_DIR" \
    --heuristic-only
  cat "$EVAL_DIR/grading_summary.json"
else
  echo "Skipping grading because RUN_GRADE=$RUN_GRADE"
fi

echo "=== KTO negative-only experiment done $(date) ==="
echo "CKPT_DIR=$CKPT_DIR"
echo "EVAL_DIR=$EVAL_DIR"
