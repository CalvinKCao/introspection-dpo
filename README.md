OLMo evaluative echo experiment

Code and recorded eval outputs for a replication study of Macar et al. (2026) on OLMo-3.1-32B, plus Echo SFT and negative-only KTO baselines.

Setup

1. Create a Python 3.11+ venv and install dependencies:
   pip install -r requirements-runpod.txt
   pip install -r requirements-grading.txt

2. Train DPO and Echo SFT adapters:
   ./runpod_paper.sh

3. Train KTO negative-only and run its first eval slice:
   ./runpod_kto_negative_500t.sh

4. Run or extend concept-injection eval:
   ./runpod_lora_500t.sh
   ./runpod_15c_extension.sh

5. Grade saved generations (heuristic, no API key):
   python grade_eval.py --heuristic-only --results results/eval/runpod-lora-40c-50t/eval_results.jsonl

6. Grade with Gemini (requires GEMINI_API_KEY or geminikey_txt.txt):
   ./run_gemini_grade_all.sh

This repository ships eval_results.jsonl and grading summary JSON files under results/eval/. LoRA adapter weights are not included; train locally or download separately.

Layout

train.py                  DPO + Echo SFT training
train_kto_negative.py     KTO negative-only training
eval_injection.py         Concept-injection eval
grade_eval.py             Heuristic and Gemini grading
gemini_judge.py           Batched Gemini LLM judge
runpod_*.sh               GPU launch scripts
EXPERIMENT.md             Full procedure and results writeup

See EXPERIMENT.md for the complete experimental description.
