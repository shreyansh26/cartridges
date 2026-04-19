from __future__ import annotations

"""Baseline evaluators for full-context prompting."""

import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cartridges.clients import VLLMClient
from cartridges.config import DEFAULT_MATRIX
from cartridges.eval.common import (
    EvalRecord,
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
    """Synchronize CUDA work so wall-clock timings line up with actual execution."""
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _clean_completion(text: str) -> str:
    """Strip reasoning markers from decoded model output before scoring."""
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    text = text.replace("<think>", " ").replace("</think>", " ")
    if text.startswith("<think>"):
        text = text.removeprefix("<think>").strip()
    return re.sub(r"\s+", " ", text).strip()


def run_vllm_quality_eval(
    *,
    eval_path: str | Path,
    output_path: str | Path,
    base_url: str,
    api_key: str,
    max_samples: int | None = None,
    max_completion_tokens: int = 128,
) -> list[EvalRecord]:
    """Evaluate the naive full-context baseline through the vLLM teacher server."""
    rows = load_eval_rows(eval_path)
    if max_samples is not None:
        rows = rows[:max_samples]

    client = VLLMClient(
        base_url=base_url,
        api_key=api_key,
        model_id=DEFAULT_MATRIX.model_id,
    )
    parity = client.probe_tokenizer_parity()
    if not parity.matches:
        raise RuntimeError(
            "Tokenizer mismatch between local HF and vLLM: "
            f"{parity.local_token_ids} != {parity.server_token_ids}"
        )

    model_config = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MATRIX.model_id,
        dtype=torch.bfloat16,
        device_map="cpu",
    ).config
    records: list[EvalRecord] = []
    try:
        for row in rows:
            messages = build_messages(row)
            prompt_text = client.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
            prompt_tokens = len(client.tokenizer.encode(prompt_text, add_special_tokens=False))

            started = time.perf_counter()
            result = client.chat(
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                temperature=0.0,
                run_mode="smoke",
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            records.append(
                EvalRecord(
                    prompt_id=f"{row['sample_id']}::{row['row_hash']}",
                    method="baseline_vllm_quality",
                    prediction=_clean_completion(result.text),
                    gold=[str(item) for item in row["answers"]],
                    exact_match=exact_match(_clean_completion(result.text), row["answers"]),
                    canonical_kv_bytes=canonical_kv_bytes(
                        num_tokens=prompt_tokens,
                        num_hidden_layers=model_config.num_hidden_layers,
                        num_key_value_heads=model_config.num_key_value_heads,
                        head_dim=_head_dim(model_config),
                    ),
                    compression_ratio=1.0,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=len(result.token_ids),
                    total_latency_ms=latency_ms,
                    metadata={
                        "sample_id": row["sample_id"],
                        "question_id": row.get("question_id"),
                        "query": row["query"],
                        "finish_reason": result.finish_reason,
                        "logprob_source": result.logprob_source,
                    },
                )
            )
    finally:
        client.close()

    write_eval_records(output_path, records)
    return records


def run_local_hf_matched_eval(
    *,
    eval_path: str | Path,
    output_path: str | Path,
    device: str,
    max_samples: int | None = None,
    max_completion_tokens: int = 128,
) -> list[EvalRecord]:
    """Evaluate the naive full-context baseline on local HF for matched timing."""
    rows = load_eval_rows(eval_path)
    if max_samples is not None:
        rows = rows[:max_samples]

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MATRIX.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MATRIX.model_id,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()

    eos_token_id = tokenizer.eos_token_id
    records: list[EvalRecord] = []
    with torch.inference_mode():
        for row in rows:
            messages = build_messages(row)
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
            encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded["input_ids"].to(device)
            prompt_tokens = input_ids.shape[-1]

            # Split timing into prefill and decode so cartridge savings are visible separately.
            _sync_if_cuda(device)
            prefill_started = time.perf_counter()
            outputs = model(input_ids=input_ids, use_cache=True)
            _sync_if_cuda(device)
            prefill_ms = (time.perf_counter() - prefill_started) * 1000.0

            generated_ids: list[int] = []
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

            _sync_if_cuda(device)
            decode_started = time.perf_counter()
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
            _sync_if_cuda(device)
            decode_seconds = time.perf_counter() - decode_started

            completion_text = _clean_completion(
                tokenizer.decode(generated_ids, skip_special_tokens=True)
            )
            total_latency_ms = prefill_ms + (decode_seconds * 1000.0)
            decode_tokens_per_second = None
            if decode_seconds > 0 and generated_ids:
                decode_tokens_per_second = len(generated_ids) / decode_seconds

            records.append(
                EvalRecord(
                    prompt_id=f"{row['sample_id']}::{row['row_hash']}",
                    method="baseline_hf_matched",
                    prediction=completion_text,
                    gold=[str(item) for item in row["answers"]],
                    exact_match=exact_match(completion_text, row["answers"]),
                    canonical_kv_bytes=canonical_kv_bytes(
                        num_tokens=prompt_tokens,
                        num_hidden_layers=model.config.num_hidden_layers,
                        num_key_value_heads=model.config.num_key_value_heads,
                        head_dim=_head_dim(model.config),
                    ),
                    compression_ratio=1.0,
                    prefill_ms=prefill_ms,
                    decode_tokens_per_second=decode_tokens_per_second,
                    total_latency_ms=total_latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=len(generated_ids),
                    metadata={
                        "sample_id": row["sample_id"],
                        "question_id": row.get("question_id"),
                        "query": row["query"],
                        "device": device,
                    },
                )
            )

    write_eval_records(output_path, records)
    return records
