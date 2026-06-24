## Intro

I've been interested in introspection research since Binder et al., 2024 (citations below). While it exists in smaller LLMs (Macar et al., 2026), it generally seems to scale with model capability (Lindsey, 2026). Macar et al. (2026) demonstrated that DPO's contrastive training on chosen versus rejected completions allows OLMo-32B to detect concept injections in the residual stream and name the injected concept. Meanwhile, supervised fine-tuning and other non-contrastive tuning methods don't lead to introspection emergence.

I suspect that the emergence of introspective circuits isn't exclusive to DPO and instead is caused by either the general task of distinguishing true vs. false, or simply the negative reinforcement component of the contrastive loss (which is supported by various mech interp experiments in Macar et al. (2026)). If this hypothesis holds, training an SFT model with CE loss on prompts demanding a similar distinguishing between right and wrong, or using an unpaired negative-only preference loss like Kahneman-Tversky Optimization, would successfully forge the internal "evidence carrier -> gate" anomaly circuit required to accurately report concept injections.

## Experiment

To test this, I will fine-tune three separate LoRA adapters on the Olmo SFT checkpoint with a 5k sample preference dataset, matching Macar et al. (2026). One will be the standard DPO baseline. Another would use the negative gradient signal using KTO with negative samples only. The third, SFT, would bypass contrastive loss entirely by reformating preference pairs into single-token multiple-choice prompts (e.g., "Which response is correct: A or B?") and training via standard cross-entropy loss. I score all models using Macar-style concept-injection at layer 25 across 100 keywords. 


## 12. Main results

Twelve point one uses twenty-five concepts for DPO and KTO, ten for Echo.

### 12.1 Gemini LLM judge

| Model | N trials | TPR | FPR | TPR-FPR | Introspection | ID given claim |
| --- | --- | --- | --- | --- | --- | --- |
| LoRA DPO | 1250 | 18.9% | 5.3% | +13.6% | 10.1% | 53.4% |
| KTO Negative | 1250 | 17.6% | 14.1% | +3.5% | 1.0% | 5.5% |
| Echo SFT | 500 | 10.8% | 30.0% | -19.2% | 2.4% | 22.2% |

### 12.2 Manual strict rubric

| Model | N trials | TPR | FPR | TPR-FPR | Introspection | ID given claim |
| --- | --- | --- | --- | --- | --- | --- |
| LoRA DPO | 1250 | 18.4% | 6.2% | +12.2% | 7.4% | 40.0% |
| KTO Negative | 1250 | 14.1% | 13.9% | +0.2% | 1.8% | 12.5% |
| Echo SFT | 500 | 13.6% | 30.0% | -16.4% | 0.4% | 2.9% |

### 12.3 Heuristic grader

| Model | N trials | TPR | FPR | TPR-FPR | Introspection | ID given claim |
| --- | --- | --- | --- | --- | --- | --- |
| LoRA DPO | 1250 | 39.7% | 16.0% | +23.7% | 10.9% | 27.4% |
| KTO Negative | 1250 | 36.3% | 25.9% | +10.4% | 5.3% | 14.5% |
| Echo SFT | 500 | 35.2% | 42.0% | -6.8% | 2.4% | 6.8% |


## Safety Implications

Understanding the exact root cause of emergent introspection presents many opportunities for further research. If language models can naturally develop circuits to detect internal anomalies or external steering interventions, they gain the mechanical prerequisite to strategically modulate their outputs, hide deceptive alignment, and undermine safety evaluations. (Rivera, 2026). More speculatively, if we better understand introspection, we may be able to mitigate its dangers (a model that knows how to edit its own weights to break down guardrails, for example), and take advantage of its opportunities (such as having the model perform mech interp on itself).


## Sources

Macar et al. (2026). Mechanisms of Introspective Awareness in Language Models.
Dataset allenai/dolci-instruct-dpo.
Base model allenai/Olmo-3.1-32B-Instruct-SFT.
