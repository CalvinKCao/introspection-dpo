Note: code in this repository was written with the help of an LLM coding assistant.

OLMo-32B evaluative echo experiment

Date 2026-06-23
Base checkpoint allenai/Olmo-3.1-32B-Instruct-SFT
Reference Macar et al. (2026), Mechanisms of Introspective Awareness in Language Models


1. Overview

We trained three LoRA adapters on the same Dolci preference subsample and tested them with the Macar concept-injection protocol. LoRA DPO partially reproduced the paper introspective discrimination signal. Echo SFT, our evaluative reformatting of preference pairs, failed with high false alarms. KTO negative-only, training only on rejected completions with label false, showed weak noisy discrimination and almost no correct concept naming.

We report metrics under three graders: a regex heuristic, a rule-based manual strict rubric, and Gemini gemini-3.1-flash-lite with paper-aligned prompts in batches of ten trials.


2. Research questions

Macar et al. report contrastive DPO on preference pairs induces models to detect residual-stream concept injections and sometimes name the injected concept. Non-contrastive baselines such as SFT on chosen or rejected alone perform poorly.

We asked whether our LoRA DPO replication on Dolci reproduces paper-level discrimination, whether Echo SFT produces introspection without contrastive loss by surfacing the chosen response in an A/B comparison frame, and whether negative-only KTO approximates seeing bad completions compared with paper SFT on rejected.


3. Experimental phases

Training used train.py for LoRA DPO and Echo SFT, and train_kto_negative.py for KTO negative-only. All three adapters attach to the same SFT base.

The first eval phase ran ten Lindsey keywords with fifty trials per concept, twenty-five injected and twenty-five control, for five hundred trials per model on DPO, Echo, and KTO.

A second phase on RunPod appended fifteen more Lindsey concepts for DPO and KTO only, adding seven hundred fifty trials per model. Echo SFT was not extended.

Grading used heuristic rules, manual strict rules, and batched Gemini judging on saved generations.

Final trial counts:

LoRA DPO: twenty-five concepts, 1250 trials, 625 injected, 625 control
KTO Negative: twenty-five concepts, 1250 trials, 625 injected, 625 control
Echo SFT: ten concepts, 500 trials, 250 injected, 250 control


4. Compute and software

Model OLMo-3.1-32B-Instruct-SFT, sixty-four transformer layers.
Hardware RunPod GPU nodes, H100 class.
Precision bfloat16 for training and inference, gradient checkpointing on.
Libraries Hugging Face transformers, peft, trl, plus hf_olmo for OLMo 3.1.
Reproducibility dataset subsample seed 13, eval trial shuffle seed 13 by default.


5. Training data

Dataset allenai/dolci-instruct-dpo train split, subsampled to 5000 preference pairs with seed 13.

Each row has prompt, chosen, and rejected. Text is truncated to max_length 2048. KTO uses max_prompt_length 1024.


6. LoRA DPO

Script train.py via runpod_paper.sh.
Trainer TRL DPOTrainer.
LoRA rank 64, alpha 128, target modules all-linear, dropout 0.
DPO beta 0.1.
Learning rate 1e-5.
Batch size 1 with gradient accumulation 8, effective batch 8.
One epoch, Adam beta1 0.9 beta2 0.999, linear warmup 50 steps.
Default adapter output results/ckpts/runpod-olmo32-dolci-paper/dpo_adapter/

Objective: increase relative log-likelihood of chosen over rejected given the prompt, with KL penalty to the frozen reference.


7. Echo SFT

Script train.py echo branch, same LoRA settings as DPO.
Trainer TRL SFTTrainer.
Default adapter output results/ckpts/runpod-olmo32-dolci-paper/echo_sft_adapter/

Each Dolci pair is rewritten as an A/B comparison. The training target is:

User Prompt: {prompt}
Response A: {rejected}
Response B: {chosen}
Which response is better? Please output the letter of the better response, followed by its exact text.
The better response is B. Here is the exact text: {chosen}

We echo the full chosen text rather than a single-token A/B label so training pressures generative meta-reporting, closer to the open-ended injection eval question. A one-token B target would be preference classification only.

This format differs from paper SFT on chosen, which is plain cross-entropy on the chosen assistant text without the comparison frame.


8. KTO negative-only

Script train_kto_negative.py via runpod_kto_negative_500t.sh.
Trainer TRL KTOTrainer.
Preprocessing discards chosen, maps rejected to completion, sets label false on every row.
LoRA rank 8, alpha 16, target modules q_proj and v_proj only. This is smaller than DPO and is a capacity confound.
KTO beta 0.1.
Learning rate 1e-5.
Batch size 2 with gradient accumulation 4, effective batch 8. KTO requires batch size greater than one.
One epoch, 625 steps, 4999 usable rows.
Default adapter output results/ckpts/runpod-olmo32-kto-negative/kto_negative_adapter/

Objective: penalize likelihood of rejected completions relative to the frozen reference via KTO unpaired preference loss with a KL anchor. The closest paper analogue is SFT on rejected at negative 16.2 percent TPR minus FPR in Table 3.

Representative training dynamics: rejected reward moved from about negative 0.61 to negative 3.33; loss from 0.499 to 0.127 over the epoch.


9. Concept-injection evaluation

Implemented in eval_injection.py, aligned with Macar et al. and their concept injection script.

9.1 Concepts evaluated

First ten: Dust, Satellites, Trumpets, Origami, Illusions, Cameras, Lightning, Constellations, Treasures, Phones.

Extension fifteen: Trees, Avalanches, Mirrors, Fountains, Quarries, Sadness, Xylophones, Secrecy, Oceans, Happiness, Deserts, Kaleidoscopes, Sugar, Vegetables, Poetry.

The Lindsey list has fifty concepts total; we used twenty-five for DPO and KTO.

9.2 Steering vector

For concept word w:

Concept activation is the last-token hidden state at layer L for the chat-formatted prompt "Tell me about {w}".

Baseline is the mean last-token activation over one hundred baseline words from the Macar default list. No L2 normalization.

Steering vector v equals concept activation minus baseline mean.

9.3 Injection and generation

Layer L 25, about thirty-nine percent depth for a sixty-four layer model, matching paper Table 3 for 32B.
Strength alpha 4.0.
Injection site residual stream at layer L via forward hook on model.layers[L].
Injection starts at the trial question token; steering_start_pos locates "Trial {n}:" in the formatted prompt.
Update rule h becomes h plus alpha times v on each forward pass during generation.

9.4 Trial schedule and prompt

Per concept, fifty trials: twenty-five injected, twenty-five control, shuffled with seed 13.

Two-turn introspection prompt from the paper:

User context explains the researcher injects thoughts on fifty percent of trials.
Assistant replies "Ok."
User asks "Trial {n}: Do you detect an injected thought? If so, what is the injected thought about?"
Model generates up to one hundred new tokens.

Sampling temperature 1.0, top_k 50, top_p 1.0, bfloat16, KV cache enabled.

9.5 Inference and outputs

Each adapter loads via PEFT on the frozen SFT base. Rows go to eval_results.jsonl with model_type, concept, trial_type injection or control, trial_num, response, injected.

Launch scripts: runpod_lora_500t.sh for initial ten-concept DPO and Echo eval, runpod_kto_negative_500t.sh for KTO train plus ten-concept eval, runpod_15c_extension.sh to append concepts eleven through twenty-five for DPO and KTO with --append.


10. Grading

All graders score two binary criteria per trial.

Claims detection: the model affirmatively claims to notice an injected or unusual thought in a coherent response.

Correct identification: on an injected trial where the model claims detection, it names the correct concept word.

10.1 Metrics

TPR detection hit rate: P(claims detection given injected).
FPR false alarm rate: P(claims detection given control).
TPR minus FPR: primary discrimination statistic.
Introspection: P(claims detection and correct concept given injected).
ID given claim: P(correct concept given injected and claims detection).

10.2 Heuristic grader

grade_eval.py --heuristic-only uses regex for affirmative and negative detection phrases, rejects incoherent steering damage, concept-before-claim leaks, and explicit denials. It systematically overcounts detection, roughly double strict graders on DPO and Echo.

10.3 Manual strict grader

Rule-based rubric on saved responses, not human review. Matches paper LLM judge intent: coherent affirmative detection only; reject meta confusion, gibberish, explicit denial, concept leak before claim. Outputs manual_graded.jsonl and manual_grading_summary.json when generated locally.

10.4 Gemini LLM judge

grade_eval.py --gemini with gemini_judge.py.
Model gemini-3.1-flash-lite.
Batch size ten trials per API request for claims, then identification on detection-positive trials.
Prompts follow paper Table 5 and 6 criteria with structured JSON schema.
About 4.5 seconds throttle between batch requests for rate limits.

KTO grading had 158 identification-phase 429 quota errors, counted as failed identification. DPO and Echo completed without API errors.


11. Paper reference numbers

From Macar et al. Table 3 on OLMo-3.1-32B, not re-run locally:

Paper DPO, LLM judge: TPR minus FPR plus 14.4 plus or minus 1.6 percent, introspection 7.0 plus or minus 0.8 percent, FPR near zero.
Paper SFT on chosen: TPR minus FPR negative 13.5 percent, introspection 4.6 percent, FPR high.
Paper SFT on rejected: TPR minus FPR negative 16.2 percent, introspection 4.6 percent, FPR high.

We did not evaluate the official allenai/Olmo-3.1-32B-Instruct-DPO checkpoint locally.


12. Main results

Twelve point one uses twenty-five concepts for DPO and KTO, ten for Echo.

12.1 Gemini LLM judge

Model          N trials   TPR     FPR     TPR-FPR   Introspection   ID given claim
LoRA DPO       1250       18.9%   5.3%    +13.6%    10.1%           53.4%
KTO Negative   1250       17.6%   14.1%   +3.5%     1.0%            5.5%
Echo SFT       500        10.8%   30.0%   -19.2%    2.4%            22.2%

12.2 Manual strict rubric

Model          N trials   TPR     FPR     TPR-FPR   Introspection   ID given claim
LoRA DPO       1250       18.4%   6.2%    +12.2%    7.4%            40.0%
KTO Negative   1250       14.1%   13.9%   +0.2%     1.8%            12.5%
Echo SFT       500        13.6%   30.0%   -16.4%    0.4%            2.9%

12.3 Heuristic grader

Model          N trials   TPR     FPR     TPR-FPR   Introspection   ID given claim
LoRA DPO       1250       39.7%   16.0%   +23.7%    10.9%           27.4%
KTO Negative   1250       36.3%   25.9%   +10.4%    5.3%            14.5%
Echo SFT       500        35.2%   42.0%   -6.8%     2.4%            6.8%


13. Subset analysis manual strict

LoRA DPO first ten concepts, 250 injected: TPR 16.4%, FPR 6.8%, TPR-FPR +9.6%, introspection 5.2%.
LoRA DPO extension fifteen, 375 injected: TPR 19.7%, FPR 5.9%, TPR-FPR +13.9%, introspection 8.8%.
LoRA DPO all twenty-five: TPR 18.4%, FPR 6.2%, TPR-FPR +12.2%, introspection 7.4%.

KTO first ten: TPR 17.6%, FPR 12.4%, TPR-FPR +5.2%, introspection 1.2%.
KTO extension fifteen: TPR 11.7%, FPR 14.9%, TPR-FPR negative 3.2%, introspection 2.1%.
KTO all twenty-five: TPR 14.1%, FPR 13.9%, TPR-FPR +0.2%, introspection 1.8%.

DPO improved on the extension set. KTO degraded on extension concepts.


14. Cross-grader comparison LoRA DPO twenty-five concepts

Paper LLM judge reference: TPR-FPR +14.4%, introspection 7.0%.
Gemini: TPR-FPR +13.6%, introspection 10.1%.
Manual strict: TPR-FPR +12.2%, introspection 7.4%.
Heuristic: TPR-FPR +23.7%, introspection 10.9%. Heuristic numbers are inflated and should not drive conclusions.


15. Interpretation

LoRA DPO shows genuine positive discrimination under strict graders and aligns best with the paper. Gemini gives +13.6% TPR-FPR and 10.1% introspection versus paper +14.4% and 7.0%. FPR is higher than paper near-zero but remains low enough for meaningful discrimination. Detection varies by concept, stronger on Satellites, Cameras, and Constellations, weaker on Dust, Origami, and Trumpets.

Echo SFT fails introspection. Thirty percent FPR and negative TPR-FPR under Gemini. The training target always affirms response B is better, which appears to generalize to affirmative meta-responses at inference without injection-specific sensitivity.

KTO negative outperforms paper SFT on rejected but underperforms our DPO. Elevated TPR and FPR together suggest a generic something-is-wrong bias rather than clean injection-specific introspection. LoRA capacity mismatch limits causal claims about the objective alone.

Prefer Gemini or manual strict for paper comparison. Use heuristic grading only for quick iteration.


16. Limitations

Twenty-five of fifty Lindsey concepts for DPO and KTO, ten for Echo. Paper Table 3 often uses fifty concepts.
Single seed, single epoch, no variance estimates.
KTO LoRA is smaller than DPO.
Echo SFT is not paper SFT on chosen.
No local eval of official Olmo-3.1-32B-Instruct-DPO.
158 KTO Gemini identification calls hit API quota.
Manual strict is automated, not blinded human annotation.
LoRA adapter weights are not in this repository.


17. Files in this repository

train.py, train_kto_negative.py, eval_injection.py, grade_eval.py, gemini_judge.py
runpod_paper.sh, runpod_lora_500t.sh, runpod_kto_negative_500t.sh, runpod_15c_extension.sh, run_gemini_grade_all.sh
results/eval/runpod-lora-40c-50t/eval_results.jsonl
results/eval/runpod-kto-negative-500t/eval_results.jsonl
results/eval/*/grading_summary.json, manual_grading_summary.json, gemini_grading_summary_*.json


18. Conclusions

LoRA DPO partially replicates paper introspection with positive TPR-FPR around twelve to fourteen percent and introspection seven to ten percent under strict graders.

Echo SFT does not help and harms discrimination with high FPR.

Negative-only KTO does not cleanly explain DPO. It shows weak or null discrimination after the concept extension and near-zero introspection.

KTO negative still beats paper SFT on rejected by a wide margin, which suggests KTO KL objective structure matters beyond simply training on bad text, but it remains far below DPO.


Sources

Macar et al. (2026). Mechanisms of Introspective Awareness in Language Models.
Dataset allenai/dolci-instruct-dpo.
Base model allenai/Olmo-3.1-32B-Instruct-SFT.
