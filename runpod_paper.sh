#!/bin/bash
# Paper-aligned DPO + Echo SFT run on a bare GPU node (e.g. RunPod H100).
#
# Usage:
#   MODEL_SIZE=7b RUN_STEM=runpod-olmo7-dolci-cache ./runpod_paper.sh
#   MODEL_SIZE=32b RUN_STEM=runpod-olmo32-dolci-paper ./runpod_paper.sh
#   MODEL_SIZE=7b RUN_TRAIN=0 RUN_STEM=runpod-06-23-5000-paper ./runpod_paper.sh
#   RUN_STEM=... ./runpod_paper.sh   # safe to rerun; resumes checkpoints

set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$PWD/hf-cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_SIZE="${MODEL_SIZE:-32b}"
case "$MODEL_SIZE" in
  7b)
    BASE_MODEL="${BASE_MODEL:-allenai/OLMo-7B-Instruct}"
    # 32B paper setting L=25 / 64 layers ~= 39%; OLMo-7B has 32 layers.
    EVAL_LAYER="${EVAL_LAYER:-12}"
    EVAL_ALPHA="${EVAL_ALPHA:-1.0}"
    DEFAULT_RUN_STEM="runpod-olmo7-dolci-cache"
    ;;
  32b)
    BASE_MODEL="${BASE_MODEL:-allenai/Olmo-3.1-32B-Instruct-SFT}"
    EVAL_LAYER="${EVAL_LAYER:-25}"
    EVAL_ALPHA="${EVAL_ALPHA:-4.0}"
    DEFAULT_RUN_STEM="runpod-olmo32-dolci-paper"
    ;;
  *)
    echo "MODEL_SIZE must be '7b' or '32b' (got '$MODEL_SIZE')" >&2
    exit 2
    ;;
esac

RUN_STEM="${RUN_STEM:-$DEFAULT_RUN_STEM}"
CKPT_DIR="results/ckpts/${RUN_STEM}"
DATA_DIR="results/datasets/${RUN_STEM}"
EVAL_DIR="${EVAL_DIR:-results/eval/${RUN_STEM}}"
LOG="results/logs/${RUN_STEM}.log"
DPO_ADAPTER="${DPO_ADAPTER:-$CKPT_DIR/dpo_adapter}"
ECHO_SFT_ADAPTER="${ECHO_SFT_ADAPTER:-$CKPT_DIR/echo_sft_adapter}"
NUM_SAMPLES="${NUM_SAMPLES:-5000}"
NUM_TRIALS="${NUM_TRIALS:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_GRADE="${RUN_GRADE:-1}"
mkdir -p results/logs "$CKPT_DIR" "$DATA_DIR" "$EVAL_DIR" "$HF_HOME"

if [[ "${SKIP_PIP_INSTALL:-0}" != "1" ]]; then
  pip install -q -r requirements-runpod.txt
fi

exec > >(tee -a "$LOG") 2>&1
echo "=== Paper-settings experiment start $(date) RUN_STEM=$RUN_STEM MODEL_SIZE=$MODEL_SIZE BASE_MODEL=$BASE_MODEL EVAL_LAYER=$EVAL_LAYER EVAL_ALPHA=$EVAL_ALPHA ==="
python -c "import torch, hf_olmo; print(torch.__version__, torch.cuda.get_device_name(0))"

if [[ "$RUN_TRAIN" == "1" ]]; then
TRAIN_EXTRA=()
if [[ "$MODEL_SIZE" == "32b" && "${LOAD_IN_4BIT:-0}" == "1" ]]; then
  TRAIN_EXTRA+=(--load-in-4bit)
fi
python -u train.py \
  --base-model "$BASE_MODEL" \
  --dataset allenai/dolci-instruct-dpo \
  --dataset-split train \
  --output-dir "$CKPT_DIR" \
  --dataset-cache-dir "$DATA_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --max-length 2048 \
  --max-prompt-length 1024 \
  --lora-r 64 \
  --lora-alpha 128 \
  --lora-dropout 0.0 \
  --target-modules all-linear \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-5 \
  --dpo-beta 0.1 \
  --warmup-steps 50 \
  --lr-scheduler-type linear \
  --adam-beta1 0.9 \
  --adam-beta2 0.999 \
  --num-train-epochs 1 \
  --save-steps 50 \
  --save-total-limit 5 \
  --save-every-minutes 60 \
  --resume \
  "${TRAIN_EXTRA[@]}"
else
  echo "Skipping training because RUN_TRAIN=$RUN_TRAIN"
fi

if [[ "$RUN_EVAL" == "1" ]]; then
python -u eval_injection.py \
  --base-model "$BASE_MODEL" \
  --dpo-adapter "$DPO_ADAPTER" \
  --echo-sft-adapter "$ECHO_SFT_ADAPTER" \
  --output-dir "$EVAL_DIR" \
  --layer "$EVAL_LAYER" \
  --alpha "$EVAL_ALPHA" \
  --num-trials "$NUM_TRIALS" \
  --max-length 2048 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature 1.0 \
  --use-cache
else
  echo "Skipping eval because RUN_EVAL=$RUN_EVAL"
fi

if [[ "$RUN_GRADE" == "1" ]]; then
python -u grade_eval.py \
  --results "$EVAL_DIR/eval_results.jsonl" \
  --output-dir "$EVAL_DIR" \
  --heuristic-only
else
  echo "Skipping grading because RUN_GRADE=$RUN_GRADE"
fi

echo "=== Paper-settings experiment done $(date) ==="
echo "CKPT_DIR=$CKPT_DIR"
echo "EVAL_DIR=$EVAL_DIR"
