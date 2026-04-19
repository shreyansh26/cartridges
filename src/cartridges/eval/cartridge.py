from __future__ import annotations

"""Matched-backend evaluation for cartridge-backed inference."""

import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cartridges.config import DEFAULT_MATRIX
from cartridges.core import TrainableKVCartridge
from cartridges.eval.common import (
    EvalRecord,
    build_cartridge_messages,
    build_messages,
    canonical_kv_bytes,
    exact_match,
    load_eval_rows,
    write_eval_records,
)


def _head_dim(model_config) -> int:
    """Infer a model's per-head KV dimension across config variants."""
    return getattr(
        model_config,
        "head_dim",
        model_config.hidden_size // model_config.num_attention_heads,
    )


def _sync_if_cuda(device: str) -> None:
    """Synchronize CUDA work so the recorded timings reflect completed kernels."""
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _clean_completion(text: str) -> str:
    """Normalize cartridge completions before scoring and semantic judging."""
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    text = text.replace("<think>", " ").replace("</think>", " ")
    if text.startswith("<think>"):
        text = text.removeprefix("<think>").strip()
    text = re.sub(r"^(?:assistant:\s*)+", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def run_cartridge_eval(
    *,
    eval_path: str | Path,
    cartridge_path: str | Path,
    output_path: str | Path,
    device: str,
    sample_id: str | None = None,
    max_samples: int | None = None,
    max_completion_tokens: int = 128,
) -> list[EvalRecord]:
    """Run local HF inference with a precomputed cartridge instead of full corpus text."""
    rows = load_eval_rows(eval_path)
    if sample_id is not None:
        rows = [row for row in rows if row["sample_id"] == sample_id]
    if max_samples is not None:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError("No evaluation rows selected for cartridge evaluation.")

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MATRIX.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MATRIX.model_id,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    cartridge = TrainableKVCartridge.load(cartridge_path, device=device)

    eos_token_id = tokenizer.eos_token_id
    records: list[EvalRecord] = []
    with torch.inference_mode():
        for row in rows:
            # Reconstruct the baseline prompt length only for byte-for-byte KV accounting.
            baseline_prompt = tokenizer.apply_chat_template(
                build_messages(row),
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
            baseline_prompt_tokens = len(
                tokenizer.encode(baseline_prompt, add_special_tokens=False)
            )
            messages = build_cartridge_messages(row)
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
            encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded["input_ids"].to(device)

            _sync_if_cuda(device)
            prefill_started = (
                torch.cuda.Event(enable_timing=True) if device.startswith("cuda") else None
            )
            prefill_ended = (
                torch.cuda.Event(enable_timing=True) if device.startswith("cuda") else None
            )
            if prefill_started is not None and prefill_ended is not None:
                prefill_started.record()
            outputs = model(
                input_ids=input_ids,
                past_key_values=cartridge.as_cache(model.config),
                use_cache=True,
            )
            if prefill_started is not None and prefill_ended is not None:
                prefill_ended.record()
                torch.cuda.synchronize(device)
                prefill_ms = float(prefill_started.elapsed_time(prefill_ended))
            else:
                prefill_ms = None

            generated_ids: list[int] = []
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

            _sync_if_cuda(device)
            decode_started = (
                torch.cuda.Event(enable_timing=True) if device.startswith("cuda") else None
            )
            decode_ended = (
                torch.cuda.Event(enable_timing=True) if device.startswith("cuda") else None
            )
            if decode_started is not None and decode_ended is not None:
                decode_started.record()
            else:
                decode_wall_start = time.perf_counter()
            for _ in range(max_completion_tokens):
                token_id = int(next_token.item())
                generated_ids.append(token_id)
                if eos_token_id is not None and token_id == eos_token_id:
                    break
                outputs = model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            if decode_started is not None and decode_ended is not None:
                decode_ended.record()
                torch.cuda.synchronize(device)
                decode_ms = float(decode_started.elapsed_time(decode_ended))
            else:
                decode_ms = (time.perf_counter() - decode_wall_start) * 1000.0

            completion_text = _clean_completion(
                tokenizer.decode(generated_ids, skip_special_tokens=True)
            )
            decode_tokens_per_second = None
            if decode_ms and decode_ms > 0 and generated_ids:
                decode_tokens_per_second = len(generated_ids) / (decode_ms / 1000.0)

            baseline_bytes = canonical_kv_bytes(
                num_tokens=baseline_prompt_tokens,
                num_hidden_layers=model.config.num_hidden_layers,
                num_key_value_heads=model.config.num_key_value_heads,
                head_dim=_head_dim(model.config),
            )
            cartridge_bytes = cartridge.canonical_kv_bytes()
            total_latency_ms = None
            if prefill_ms is not None and decode_ms is not None:
                total_latency_ms = prefill_ms + decode_ms

            records.append(
                EvalRecord(
                    prompt_id=f"{row['sample_id']}::{row['row_hash']}",
                    method="cartridge_hf_matched",
                    prediction=completion_text,
                    gold=[str(item) for item in row["answers"]],
                    exact_match=exact_match(completion_text, row["answers"]),
                    canonical_kv_bytes=cartridge_bytes,
                    compression_ratio=baseline_bytes / cartridge_bytes,
                    prefill_ms=prefill_ms,
                    decode_tokens_per_second=decode_tokens_per_second,
                    total_latency_ms=total_latency_ms,
                    prompt_tokens=int(input_ids.shape[-1]),
                    completion_tokens=len(generated_ids),
                    metadata={
                        "sample_id": row["sample_id"],
                        "question_id": row.get("question_id"),
                        "query": row["query"],
                        "baseline_canonical_kv_bytes": baseline_bytes,
                        "device": device,
                        "cartridge_path": str(Path(cartridge_path).resolve()),
                    },
                )
            )

    write_eval_records(output_path, records)
    return records
