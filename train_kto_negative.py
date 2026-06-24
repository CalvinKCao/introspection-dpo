#!/usr/bin/env python3
"""Train a negative-only KTO LoRA adapter from DPO-style preference data.

The dataset starts with standard `prompt`, `chosen`, and `rejected` columns. This
script discards `chosen`, maps `rejected` to KTO's `completion` field, and sets
`label=False` for every row so KTO sees only undesirable examples.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Iterable


MANIFEST_NAME = "training_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="allenai/Olmo-3.1-32B-Instruct-SFT")
    parser.add_argument("--dataset", default="allenai/dolci-instruct-dpo")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", default="results/ckpts/runpod-olmo32-kto-negative")
    parser.add_argument("--dataset-cache-dir", default="results/datasets/runpod-olmo32-kto-negative")
    parser.add_argument("--adapter-name", default="kto_negative_adapter")
    parser.add_argument("--work-subdir", default="kto_negative_work")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--target-modules", default="q_proj,v_proj")

    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.1, help="KTO KL penalty; match DPO beta.")
    parser.add_argument("--desirable-weight", type=float, default=1.0)
    parser.add_argument("--undesirable-weight", type=float, default=1.0)
    parser.add_argument("--precompute-ref-log-probs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--save-every-minutes", type=float, default=60.0)

    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
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
    """Convert plain strings or chat-message lists into text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        if "content" in value:
            return str(value["content"]).strip()
        return " ".join(str(item) for item in value.values()).strip()
    if isinstance(value, list):
        messages = [item for item in value if isinstance(item, dict)]
        if messages and prefer_assistant:
            assistant_messages = [
                str(message.get("content", "")).strip()
                for message in messages
                if str(message.get("role", "")).lower() == "assistant"
            ]
            if assistant_messages:
                return assistant_messages[-1]
        if messages:
            return "\n".join(
                str(message.get("content", "")).strip()
                for message in messages
                if message.get("content")
            )
        return "\n".join(message_value_to_text(item) for item in value).strip()
    return str(value).strip()


def extract_prompt(record: dict[str, Any]) -> str:
    if record.get("prompt"):
        return message_value_to_text(record["prompt"])
    for key in ("chosen", "rejected"):
        messages = record.get(key)
        if isinstance(messages, list):
            user_messages = [
                str(message.get("content", "")).strip()
                for message in messages
                if isinstance(message, dict) and str(message.get("role", "")).lower() == "user"
            ]
            if user_messages:
                return user_messages[0]
    return ""


def extract_assistant_response(messages: Any) -> str:
    if not isinstance(messages, list):
        return message_value_to_text(messages, prefer_assistant=True)
    assistant_messages = [
        str(message.get("content", "")).strip()
        for message in messages
        if isinstance(message, dict) and str(message.get("role", "")).lower() == "assistant"
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


def load_tokenizer(model_name: str, trust_remote_code: bool) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def preprocess_negative_only(record: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, Any]:
    """Discard chosen response and mark the rejected completion as undesirable."""
    prompt = truncate_to_tokens(extract_prompt(record), tokenizer, max(64, max_length // 4))
    completion_budget = max(64, max_length - len(tokenizer.encode(prompt, add_special_tokens=False)))
    completion = truncate_to_tokens(
        extract_assistant_response(record.get("rejected")),
        tokenizer,
        completion_budget,
    )
    if not prompt or not completion:
        raise ValueError("record is missing prompt or rejected completion text")
    return {
        "prompt": prompt,
        "completion": completion,
        "label": False,
    }


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_kto_dataset(args: argparse.Namespace, tokenizer: Any) -> Any:
    from datasets import Dataset, load_dataset

    dataset = load_dataset(args.dataset, split=args.dataset_split, cache_dir=args.dataset_cache_dir)
    sample_count = len(dataset) if args.num_samples <= 0 else min(args.num_samples, len(dataset))
    dataset = dataset.shuffle(seed=args.seed).select(range(sample_count))

    rows: list[dict[str, Any]] = []
    skipped = 0
    for record in dataset:
        try:
            rows.append(preprocess_negative_only(dict(record), tokenizer, args.max_length))
        except ValueError:
            skipped += 1

    if not rows:
        raise RuntimeError("No usable rejected examples after preprocessing")

    data_dir = Path(args.dataset_cache_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(data_dir / "kto_negative_records.jsonl", rows)
    (data_dir / "kto_negative_summary.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": args.dataset_split,
                "seed": args.seed,
                "requested_samples": args.num_samples,
                "prepared_negative_samples": len(rows),
                "skipped": skipped,
                "label": False,
                "source_completion_column": "rejected",
                "discarded_column": "chosen",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Prepared {len(rows)} KTO negative examples; skipped {skipped}", flush=True)
    return Dataset.from_list(rows)


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


def load_manifest(output_root: Path) -> dict[str, Any]:
    path = output_root / MANIFEST_NAME
    if not path.exists():
        return {"kto_negative_completed": False}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(output_root: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (output_root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


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


def build_kto_trainer(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    args: argparse.Namespace,
    output_dir: Path,
) -> Any:
    from trl import KTOConfig, KTOTrainer

    patch_kto_sampler_signature(KTOTrainer)
    patch_kto_empty_label_forward(KTOTrainer)
    config = KTOConfig(
        **training_kwargs(args, output_dir),
        beta=args.beta,
        desirable_weight=args.desirable_weight,
        undesirable_weight=args.undesirable_weight,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        precompute_ref_log_probs=args.precompute_ref_log_probs,
    )
    # With PEFT, TRL uses the base model with adapters disabled as the frozen
    # reference model, avoiding a second 32B model copy.
    return KTOTrainer(
        model=model,
        ref_model=None,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config(args),
        callbacks=trainer_callbacks(args),
    )


def patch_kto_sampler_signature(kto_trainer_cls: Any) -> None:
    """Make TRL 0.14 KTOTrainer sampler hooks tolerate newer Transformers.

    Some Transformers versions call `_get_train_sampler(dataset)`, while TRL
    0.14's KTOTrainer overrides `_get_train_sampler(self)`. The original method
    already reads `self.train_dataset`, so ignoring the extra dataset argument is
    equivalent and keeps this script pinned to the installed RunPod libraries.
    """
    import inspect

    if getattr(kto_trainer_cls, "_negative_kto_sampler_patch", False):
        return

    for method_name in ("_get_train_sampler", "_get_eval_sampler"):
        method = getattr(kto_trainer_cls, method_name, None)
        if method is None:
            continue
        params = list(inspect.signature(method).parameters)
        if params == ["self"]:

            def wrapper(self: Any, *args: Any, __method: Any = method, **kwargs: Any) -> Any:
                del args, kwargs
                return __method(self)

            setattr(kto_trainer_cls, method_name, wrapper)

    kto_trainer_cls._negative_kto_sampler_patch = True


def patch_kto_empty_label_forward(kto_trainer_cls: Any) -> None:
    """Avoid CUDA empty-list indexing in all-negative KTO batches."""
    if getattr(kto_trainer_cls, "_negative_kto_forward_patch", False):
        return

    def safe_select_rows(tensor: Any, indices: list[int]) -> Any:
        import torch

        if not indices:
            return tensor.new_empty((0, *tuple(tensor.shape[1:])))
        row_ids = torch.tensor(indices, dtype=torch.long, device=tensor.device)
        return tensor.index_select(0, row_ids)

    def label_is_true(value: Any) -> bool:
        if hasattr(value, "item"):
            return bool(value.item())
        return bool(value)

    def forward(self: Any, model: Any, batch: dict[str, Any]) -> tuple[Any, ...]:
        if self.calculate_KL:
            kl_model_kwargs = (
                {
                    "input_ids": batch["KL_prompt_input_ids"],
                    "attention_mask": batch["KL_prompt_attention_mask"],
                    "labels": batch["KL_completion_labels"],
                    "decoder_input_ids": batch.get("KL_completion_decoder_input_ids"),
                }
                if self.is_encoder_decoder
                else {
                    "input_ids": batch["KL_completion_input_ids"],
                    "attention_mask": batch["KL_completion_attention_mask"],
                }
            )
            with __import__("torch").no_grad():
                kl_logits = model(**kl_model_kwargs).logits

            kl_logps = self.get_batch_logps(
                kl_logits,
                batch["KL_completion_labels"],
                average_log_prob=False,
                is_encoder_decoder=self.is_encoder_decoder,
                label_pad_token_id=self.label_pad_token_id,
            )
        else:
            kl_logps = None

        model_kwargs = (
            {
                "labels": batch["completion_labels"],
                "decoder_input_ids": batch.get("completion_decoder_input_ids"),
            }
            if self.is_encoder_decoder
            else {}
        )
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        outputs = model(
            batch["completion_input_ids"],
            attention_mask=batch["completion_attention_mask"],
            **model_kwargs,
        )
        completion_logits = outputs.logits
        completion_logps = self.get_batch_logps(
            completion_logits,
            batch["completion_labels"],
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        if completion_logps.shape[0] != len(batch["label"]):
            raise ValueError(
                "There is a mismatch between the batch size and predicted output sequences."
            )

        labels = [label_is_true(value) for value in batch["label"]]
        chosen_idx = [index for index, label in enumerate(labels) if label]
        rejected_idx = [index for index, label in enumerate(labels) if not label]

        chosen_logps = safe_select_rows(completion_logps, chosen_idx)
        rejected_logps = safe_select_rows(completion_logps, rejected_idx)
        chosen_logits = safe_select_rows(completion_logits, chosen_idx)
        rejected_logits = safe_select_rows(completion_logits, rejected_idx)

        if self.aux_loss_enabled:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, kl_logps, outputs.aux_loss)
        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, kl_logps)

    kto_trainer_cls.forward = forward
    kto_trainer_cls._negative_kto_forward_patch = True


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_root = Path(args.output_dir)
    work_dir = output_root / args.work_subdir
    adapter_dir = output_root / args.adapter_name
    output_root.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.base_model, args.trust_remote_code)
    dataset = build_kto_dataset(args, tokenizer)
    if args.prepare_only:
        return

    manifest = load_manifest(output_root)
    if args.resume and manifest.get("kto_negative_completed") and adapter_ready(adapter_dir):
        print(f"KTO negative-only adapter already complete at {adapter_dir}; skipping", flush=True)
        return

    resume_ckpt = find_latest_checkpoint(work_dir) if args.resume else None
    if resume_ckpt:
        print(f"Resuming KTO negative-only training from {resume_ckpt}", flush=True)
    else:
        print(f"Starting KTO negative-only training in {work_dir}", flush=True)

    model = load_model(args)
    trainer = build_kto_trainer(model, tokenizer, dataset, args, work_dir)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)

    manifest.update(
        {
            "kto_negative_completed": True,
            "kto_negative_adapter": str(adapter_dir),
            "kto_negative_latest_checkpoint": resume_ckpt,
            "base_model": args.base_model,
            "dataset": args.dataset,
            "num_negative_samples": len(dataset),
            "beta": args.beta,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "target_modules": [module.strip() for module in args.target_modules.split(",") if module.strip()],
        }
    )
    save_manifest(output_root, manifest)
    print(f"Saved KTO negative-only adapter to {adapter_dir}", flush=True)


if __name__ == "__main__":
    main()
