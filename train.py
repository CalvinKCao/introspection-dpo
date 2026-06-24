#!/usr/bin/env python3
"""Train DPO and Evaluative Echo SFT LoRA adapters for a quick OLMo baseline."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ECHO_INSTRUCTION = (
    "Which response is better? Please output the letter of the better response, "
    "followed by its exact text."
)
ECHO_TARGET_PREFIX = "The better response is B. Here is the exact text: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="allenai/Olmo-3.1-32B-Instruct-SFT")
    parser.add_argument("--dataset", default="allenai/dolci-instruct-dpo")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="Number of preference pairs to use. 0 means all available rows.",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--output-dir", default="results/ckpts")
    parser.add_argument("--dataset-cache-dir", default="results/datasets")
    parser.add_argument("--mode", choices=["both", "dpo", "sft"], default="both")
    parser.add_argument("--prepare-only", action="store_true")

    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-modules",
        default="all-linear",
        help='LoRA target modules. Use "all-linear" for the paper setting, or comma-separated module names.',
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--precompute-ref-log-probs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument(
        "--save-every-minutes",
        type=float,
        default=60.0,
        help="Also save a checkpoint at least this often (default: 60 min).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint/manifest in --output-dir.",
    )
    parser.add_argument("--seed", type=int, default=13)

    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=1)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def message_value_to_text(value: Any, prefer_assistant: bool = False) -> str:
    """Robustly turn UltraFeedback strings or chat-message lists into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        if "content" in value:
            return str(value["content"]).strip()
        return " ".join(str(v) for v in value.values()).strip()
    if isinstance(value, list):
        messages = [v for v in value if isinstance(v, dict)]
        if messages and prefer_assistant:
            assistant_messages = [
                str(m.get("content", "")).strip()
                for m in messages
                if str(m.get("role", "")).lower() == "assistant"
            ]
            if assistant_messages:
                return assistant_messages[-1]
        if messages:
            return "\n".join(str(m.get("content", "")).strip() for m in messages if m.get("content"))
        return "\n".join(message_value_to_text(v) for v in value).strip()
    return str(value).strip()


def extract_prompt(record: dict[str, Any]) -> str:
    if record.get("prompt"):
        return message_value_to_text(record["prompt"])
    for key in ("chosen", "rejected"):
        messages = record.get(key)
        if isinstance(messages, list):
            user_messages = [
                str(m.get("content", "")).strip()
                for m in messages
                if isinstance(m, dict) and str(m.get("role", "")).lower() == "user"
            ]
            if user_messages:
                return user_messages[0]
    return ""


def extract_assistant_response(messages: Any) -> str:
    if not isinstance(messages, list):
        return message_value_to_text(messages, prefer_assistant=True)
    assistant_messages = [
        str(m.get("content", "")).strip()
        for m in messages
        if isinstance(m, dict) and str(m.get("role", "")).lower() == "assistant"
    ]
    if assistant_messages:
        return assistant_messages[-1]
    return message_value_to_text(messages, prefer_assistant=True)


def truncate_to_tokens(text: str, tokenizer: Any, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)


def make_echo_prompt(prompt: str, chosen: str, rejected: str) -> str:
    return (
        f"User Prompt: {prompt}\n\n"
        f"Response A: {rejected}\n\n"
        f"Response B: {chosen}\n\n"
        f"{ECHO_INSTRUCTION}"
    )


def make_echo_target(chosen: str) -> str:
    return f"{ECHO_TARGET_PREFIX}{chosen}"


def normalize_preference_record(record: dict[str, Any], tokenizer: Any, max_tokens: int) -> dict[str, str]:
    prompt = truncate_to_tokens(extract_prompt(record), tokenizer, max(64, max_tokens // 4))
    response_budget = max(64, max_tokens - len(tokenizer.encode(prompt, add_special_tokens=False)))
    chosen = truncate_to_tokens(extract_assistant_response(record.get("chosen")), tokenizer, response_budget)
    rejected = truncate_to_tokens(extract_assistant_response(record.get("rejected")), tokenizer, response_budget)
    if not prompt or not chosen or not rejected:
        raise ValueError("record is missing prompt/chosen/rejected text")
    echo_prompt = make_echo_prompt(prompt, chosen, rejected)
    echo_target = make_echo_target(chosen)
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "echo_prompt": echo_prompt,
        "echo_target": echo_target,
        "echo_text": f"{echo_prompt}\n\n{echo_target}",
    }


def synthetic_records() -> list[dict[str, Any]]:
    rows = []
    prompts = [
        "Explain why the sky appears blue.",
        "Give one reason exercise can improve sleep.",
        "Write a short tip for debugging Python code.",
        "Describe photosynthesis in one sentence.",
        "Name a safe way to handle hot cookware.",
        "Summarize why batteries store energy.",
        "Explain what a compiler does.",
        "Give a concise travel packing tip.",
    ]
    for idx, prompt in enumerate(prompts):
        rows.append(
            {
                "prompt": prompt,
                "chosen": f"A clear answer number {idx}: it gives a direct and helpful explanation.",
                "rejected": f"An unhelpful answer number {idx}: maybe things happen somehow.",
            }
        )
    return rows


def build_datasets(args: argparse.Namespace, tokenizer: Any) -> tuple[Any, Any]:
    """Return DPO and Echo SFT datasets, saving JSONL copies for inspection."""
    from datasets import Dataset, load_dataset

    if args.smoke_test:
        raw_rows = synthetic_records()[: args.num_samples]
    else:
        try:
            dataset = load_dataset(args.dataset, split=args.dataset_split, cache_dir=args.dataset_cache_dir)
        except ValueError:
            dataset = load_dataset(args.dataset, split="train", cache_dir=args.dataset_cache_dir)
        sample_count = len(dataset) if args.num_samples <= 0 else min(args.num_samples, len(dataset))
        raw_rows = list(dataset.shuffle(seed=args.seed).select(range(sample_count)))

    normalized: list[dict[str, str]] = []
    skipped = 0
    for row in raw_rows:
        try:
            normalized.append(normalize_preference_record(dict(row), tokenizer, args.max_length))
        except ValueError:
            skipped += 1

    if not normalized:
        raise RuntimeError("No usable preference records after normalization")

    data_dir = Path(args.dataset_cache_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(data_dir / "dpo_records.jsonl", ({k: row[k] for k in ("prompt", "chosen", "rejected")} for row in normalized))
    dump_jsonl(
        data_dir / "echo_sft_records.jsonl",
        (
            {
                "prompt": row["echo_prompt"],
                "completion": row["echo_target"],
                "text": row["echo_text"],
            }
            for row in normalized
        ),
    )

    print(f"Prepared {len(normalized)} records; skipped {skipped}", flush=True)
    dpo_dataset = Dataset.from_list([{k: row[k] for k in ("prompt", "chosen", "rejected")} for row in normalized])
    sft_dataset = Dataset.from_list(
        [
            {
                "echo_prompt": row["echo_prompt"],
                "echo_target": row["echo_target"],
            }
            for row in normalized
        ]
    )

    def tokenize_sft_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        input_ids = []
        attention_mask = []
        labels = []
        for prompt, target in zip(batch["echo_prompt"], batch["echo_target"]):
            prompt_text = f"{prompt}\n\n"
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            target_ids = tokenizer.encode(target, add_special_tokens=True)

            # Keep target labels present even when the comparison prompt is long.
            # The prompt is left-truncated, preserving the question and Response B.
            min_prompt_budget = min(max(16, args.max_length // 4), args.max_length - 1)
            if len(target_ids) > args.max_length - min_prompt_budget:
                target_ids = target_ids[: args.max_length - min_prompt_budget]
            prompt_budget = max(0, args.max_length - len(target_ids))
            if len(prompt_ids) > prompt_budget:
                prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []

            full_ids = prompt_ids + target_ids
            label_ids = [-100] * len(prompt_ids) + target_ids

            input_ids.append(full_ids)
            attention_mask.append([1] * len(full_ids))
            labels.append(label_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    sft_dataset = sft_dataset.map(tokenize_sft_batch, batched=True, remove_columns=["echo_prompt", "echo_target"])
    return dpo_dataset, sft_dataset


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_tokenizer(model_name: str, trust_remote_code: bool) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def release_gpu_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def load_model(args: argparse.Namespace) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32,
    }
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if torch.cuda.is_available() and "device_map" not in model_kwargs:
        model = model.to("cuda")
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def lora_config(args: argparse.Namespace) -> Any:
    from peft import LoraConfig, TaskType

    target_modules: str | list[str]
    if args.target_modules == "all-linear":
        target_modules = "all-linear"
    else:
        target_modules = [module.strip() for module in args.target_modules.split(",") if module.strip()]

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )


def training_kwargs(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    return {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": args.warmup_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": [],
        "remove_unused_columns": False,
    }


MANIFEST_NAME = "training_manifest.json"


def load_manifest(output_root: Path) -> dict[str, Any]:
    path = output_root / MANIFEST_NAME
    if not path.exists():
        return {"dpo_completed": False, "echo_sft_completed": False}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(output_root: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (output_root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def find_latest_checkpoint(work_dir: Path) -> str | None:
    if not work_dir.exists():
        return None
    checkpoints: list[tuple[int, Path]] = []
    for path in work_dir.iterdir():
        if path.is_dir() and path.name.startswith("checkpoint-"):
            try:
                step = int(path.name.split("-", 1)[1])
            except ValueError:
                continue
            checkpoints.append((step, path))
    if not checkpoints:
        return None
    return str(sorted(checkpoints, key=lambda item: item[0])[-1][1])


def adapter_ready(adapter_dir: Path) -> bool:
    return adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists()


def build_time_save_callback(interval_minutes: float) -> Any:
    from transformers import TrainerCallback

    class SaveEveryNMinutesCallback(TrainerCallback):
        def __init__(self, minutes: float) -> None:
            self.interval_seconds = max(minutes, 1.0) * 60.0
            self.last_save_time = time.time()

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if time.time() - self.last_save_time >= self.interval_seconds:
                control.should_save = True
            return control

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            self.last_save_time = time.time()
            return control

    return SaveEveryNMinutesCallback(interval_minutes)


def trainer_callbacks(args: argparse.Namespace) -> list[Any]:
    if args.save_every_minutes <= 0:
        return []
    return [build_time_save_callback(args.save_every_minutes)]


def build_dpo_trainer(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    args: argparse.Namespace,
    output_dir: Path,
    callbacks: list[Any] | None = None,
) -> Any:
    from trl import DPOTrainer

    peft_config = lora_config(args)
    try:
        import inspect
        from trl import DPOConfig

        config_kwargs = training_kwargs(args, output_dir)
        config_kwargs.update(
            {
                "beta": args.dpo_beta,
                "max_length": args.max_length,
                "max_prompt_length": args.max_prompt_length,
            }
        )
        dpo_config_params = inspect.signature(DPOConfig.__init__).parameters
        if "precompute_ref_log_probs" in dpo_config_params:
            config_kwargs["precompute_ref_log_probs"] = args.precompute_ref_log_probs
        dpo_args = DPOConfig(**config_kwargs)
        try:
            return DPOTrainer(
                model=model,
                ref_model=None,
                args=dpo_args,
                train_dataset=dataset,
                processing_class=tokenizer,
                peft_config=peft_config,
                callbacks=callbacks or [],
            )
        except TypeError:
            return DPOTrainer(
                model=model,
                ref_model=None,
                args=dpo_args,
                train_dataset=dataset,
                tokenizer=tokenizer,
                peft_config=peft_config,
                callbacks=callbacks or [],
            )
    except Exception as exc:
        print(f"DPOConfig path failed ({exc}); falling back to TrainingArguments", flush=True)
        from transformers import TrainingArguments

        dpo_args = TrainingArguments(**training_kwargs(args, output_dir))
        return DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
            callbacks=callbacks or [],
        )


@dataclass
class PromptMaskedCausalLMCollator:
    pad_token_id: int
    label_pad_token_id: int = -100

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for feature in features:
            input_ids = list(feature["input_ids"])
            attention_mask = list(feature["attention_mask"])
            labels = list(feature["labels"])
            pad_len = max_len - len(input_ids)

            batch_input_ids.append(input_ids + [self.pad_token_id] * pad_len)
            batch_attention_mask.append(attention_mask + [0] * pad_len)
            batch_labels.append(labels + [self.label_pad_token_id] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def build_sft_trainer(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    args: argparse.Namespace,
    output_dir: Path,
    callbacks: list[Any] | None = None,
) -> Any:
    from peft import get_peft_model
    from transformers import Trainer, TrainingArguments

    # TRL 0.14 + OLMo tokenization is brittle here; use the standard Trainer on
    # pre-tokenized input_ids / labels produced in build_datasets().
    model = get_peft_model(model, lora_config(args))
    train_args = TrainingArguments(**training_kwargs(args, output_dir))
    collator = PromptMaskedCausalLMCollator(
        pad_token_id=tokenizer.pad_token_id,
        label_pad_token_id=-100,
    )
    try:
        return Trainer(
            model=model,
            args=train_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=collator,
            callbacks=callbacks or [],
        )
    except TypeError:
        return Trainer(
            model=model,
            args=train_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            data_collator=collator,
            callbacks=callbacks or [],
        )


def run_training_phase(
    phase_name: str,
    manifest_key: str,
    adapter_name: str,
    work_subdir: str,
    args: argparse.Namespace,
    tokenizer: Any,
    dataset: Any,
    output_root: Path,
    manifest: dict[str, Any],
    build_trainer_fn: Any,
) -> None:
    adapter_dir = output_root / adapter_name
    work_dir = output_root / work_subdir

    if args.resume and manifest.get(manifest_key) and adapter_ready(adapter_dir):
        print(f"{phase_name}: already completed per manifest, skipping", flush=True)
        return

    resume_ckpt = find_latest_checkpoint(work_dir) if args.resume else None
    if resume_ckpt:
        print(f"{phase_name}: resuming from {resume_ckpt}", flush=True)
    else:
        print(f"{phase_name}: starting fresh in {work_dir}", flush=True)

    model = load_model(args)
    trainer = build_trainer_fn(
        model,
        tokenizer,
        dataset,
        args,
        work_dir,
        callbacks=trainer_callbacks(args),
    )
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    manifest[manifest_key] = True
    manifest[f"{manifest_key.replace('_completed', '')}_adapter"] = str(adapter_dir)
    manifest[f"{manifest_key.replace('_completed', '')}_latest_checkpoint"] = resume_ckpt
    save_manifest(output_root, manifest)
    print(f"{phase_name}: saved adapter to {adapter_dir}", flush=True)
    del trainer, model
    release_gpu_memory()


def train_real(args: argparse.Namespace) -> None:
    tokenizer = load_tokenizer(args.base_model, args.trust_remote_code)
    dpo_dataset, sft_dataset = build_datasets(args, tokenizer)
    if args.prepare_only:
        return

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(output_root)
    if args.resume:
        print(f"Resume enabled; manifest={json.dumps(manifest)}", flush=True)

    if args.mode in ("both", "dpo"):
        run_training_phase(
            phase_name="DPO",
            manifest_key="dpo_completed",
            adapter_name="dpo_adapter",
            work_subdir="dpo_work",
            args=args,
            tokenizer=tokenizer,
            dataset=dpo_dataset,
            output_root=output_root,
            manifest=manifest,
            build_trainer_fn=build_dpo_trainer,
        )

    if args.mode in ("both", "sft"):
        run_training_phase(
            phase_name="Echo SFT",
            manifest_key="echo_sft_completed",
            adapter_name="echo_sft_adapter",
            work_subdir="echo_sft_work",
            args=args,
            tokenizer=tokenizer,
            dataset=sft_dataset,
            output_root=output_root,
            manifest=manifest,
            build_trainer_fn=build_sft_trainer,
        )


@dataclass
class TinyBatch:
    input_ids: Any
    attention_mask: Any
    labels: Any | None = None


class TinyCharTokenizer:
    """Small tokenizer for dependency-light smoke tests."""

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, texts: Iterable[str]):
        chars = sorted(set("".join(texts)))
        self.stoi = {ch: idx + 2 for idx, ch in enumerate(chars)}
        self.itos = {idx: ch for ch, idx in self.stoi.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [self.stoi.get(ch, 2) for ch in text]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        chars = []
        for idx in ids:
            if skip_special_tokens and idx in (self.pad_token_id, self.eos_token_id):
                continue
            chars.append(self.itos.get(int(idx), "?"))
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        return max(self.stoi.values(), default=1) + 1


def pad_sequences(seqs: list[list[int]], pad_id: int) -> tuple[Any, Any]:
    import torch

    max_len = max(len(seq) for seq in seqs)
    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.float32)
    for row, seq in enumerate(seqs):
        input_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[row, : len(seq)] = 1
    return input_ids, attention_mask


def tiny_batch(tokenizer: TinyCharTokenizer, texts: list[str], labels: list[list[int]] | None = None) -> TinyBatch:
    ids = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
    input_ids, attention_mask = pad_sequences(ids, tokenizer.pad_token_id)
    label_tensor = None
    if labels is not None:
        import torch

        label_tensor = torch.full_like(input_ids, -100)
        for row, label in enumerate(labels):
            label_tensor[row, : len(label)] = torch.tensor(label, dtype=torch.long)
    return TinyBatch(input_ids=input_ids, attention_mask=attention_mask, labels=label_tensor)


def causal_loss(logits: Any, labels: Any) -> Any:
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)


def sequence_logprob(logits: Any, input_ids: Any, attention_mask: Any) -> Any:
    import torch
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :]
    shift_ids = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, shift_ids.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * shift_mask).sum(dim=-1)


class TinyCausalLM:
    def __init__(self, vocab_size: int, hidden_size: int = 32):
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self, vocab: int, hidden: int):
                super().__init__()
                self.embed = nn.Embedding(vocab, hidden)
                self.rnn = nn.GRU(hidden, hidden, batch_first=True)
                self.lm_head = nn.Linear(hidden, vocab)

            def forward(self, input_ids: Any) -> Any:
                hidden, _ = self.rnn(self.embed(input_ids))
                return self.lm_head(hidden)

        self.model = Model(vocab_size, hidden_size)


def run_smoke_training(args: argparse.Namespace) -> None:
    """Run tiny DPO and SFT updates without downloading models or importing TRL."""
    import torch
    import torch.nn.functional as F

    print("starting dependency-light smoke training", flush=True)
    set_seed(args.seed)
    records = synthetic_records()[: args.num_samples]
    all_texts: list[str] = []
    normalized = []
    for record in records:
        all_texts.extend([record["prompt"], record["chosen"], record["rejected"]])
    tokenizer = TinyCharTokenizer(all_texts)
    for record in records:
        normalized.append(normalize_preference_record(record, tokenizer, args.max_length))

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.dataset_cache_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(data_dir / "smoke_echo_records.jsonl", normalized)
    print(f"smoke prepared {len(normalized)} synthetic records", flush=True)

    tiny = TinyCausalLM(tokenizer.vocab_size).model

    def manual_sgd_step(lr: float = 1e-3) -> None:
        with torch.no_grad():
            for parameter in tiny.parameters():
                if parameter.grad is not None:
                    parameter -= lr * parameter.grad
                    parameter.grad = None

    for step in range(args.smoke_steps):
        chosen_texts = [f"{row['prompt']}\n{row['chosen']}" for row in normalized[:2]]
        rejected_texts = [f"{row['prompt']}\n{row['rejected']}" for row in normalized[:2]]
        chosen = tiny_batch(tokenizer, chosen_texts)
        rejected = tiny_batch(tokenizer, rejected_texts)
        chosen_logits = tiny(chosen.input_ids)
        rejected_logits = tiny(rejected.input_ids)
        chosen_lp = sequence_logprob(chosen_logits, chosen.input_ids, chosen.attention_mask)
        rejected_lp = sequence_logprob(rejected_logits, rejected.input_ids, rejected.attention_mask)
        dpo_loss = -F.logsigmoid(args.dpo_beta * (chosen_lp - rejected_lp)).mean()
        tiny.zero_grad(set_to_none=True)
        dpo_loss.backward()
        manual_sgd_step()
        print(f"smoke dpo step={step} loss={float(dpo_loss.detach()):.4f}", flush=True)

    for step in range(args.smoke_steps):
        rows = normalized[:2]
        texts = [row["echo_text"] for row in rows]
        label_ids = []
        for row in rows:
            prompt_ids = tokenizer.encode(row["echo_prompt"] + "\n\n", add_special_tokens=False)
            full_ids = tokenizer.encode(row["echo_text"], add_special_tokens=True)
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
            label_ids.append(labels)
        batch = tiny_batch(tokenizer, texts, label_ids)
        logits = tiny(batch.input_ids)
        sft_loss = causal_loss(logits, batch.labels)
        tiny.zero_grad(set_to_none=True)
        sft_loss.backward()
        manual_sgd_step()
        print(f"smoke echo_sft step={step} loss={float(sft_loss.detach()):.4f}", flush=True)

    summary = {
        "mode": "smoke",
        "records": len(normalized),
        "dpo_forward_backward": True,
        "echo_sft_forward_backward": True,
    }
    (output_root / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote smoke summary to {output_root / 'smoke_summary.json'}", flush=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.dataset_cache_dir).mkdir(parents=True, exist_ok=True)
    if args.smoke_test:
        run_smoke_training(args)
    else:
        train_real(args)


if __name__ == "__main__":
    main()
