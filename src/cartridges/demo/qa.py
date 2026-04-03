from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cartridges.config import DEFAULT_MATRIX
from cartridges.core import TrainableKVCartridge
from cartridges.data.arxiv_smoke import build_arxiv_smoke_manifest
from cartridges.data.common import write_json
from cartridges.eval.common import (
    CARTRIDGE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    canonical_kv_bytes,
)
from cartridges.synthesis import run_self_study_synthesis
from cartridges.train import train_cartridge


def _head_dim(model_config) -> int:
    return getattr(
        model_config,
        "head_dim",
        model_config.hidden_size // model_config.num_attention_heads,
    )


def _sync_if_cuda(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _decode_visible_answer(tokenizer, generated_ids: list[int]) -> str:
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    if text.startswith("<think>"):
        text = text.removeprefix("<think>").strip()
    return re.sub(r"\s+", " ", text).strip()


def build_demo_cartridge(
    *,
    source_path: str | Path,
    work_dir: str | Path,
    base_url: str,
    api_key: str,
    device: str,
    chunk_tokens: int = 1024,
    stride_tokens: int | None = None,
    synthesis_num_samples: int = 8,
    synthesis_top_logprobs: int = 5,
    cartridge_tokens: int = 256,
    train_steps: int = 30,
    synthesis_max_completion_tokens_a: int = 64,
    synthesis_max_completion_tokens_b: int = 192,
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    stride_tokens = stride_tokens or chunk_tokens

    corpus_manifest_path = work_dir / "corpus_manifest.json"
    corpus_manifest = build_arxiv_smoke_manifest(
        source_path=source_path,
        output_path=corpus_manifest_path,
        chunk_tokens=chunk_tokens,
        stride_tokens=stride_tokens,
    )
    if corpus_manifest["num_chunks"] != 1:
        raise ValueError(
            "The demo workflow expects a single chunk so one cartridge represents the full corpus. "
            f"Current settings produced {corpus_manifest['num_chunks']} chunks. Increase "
            "`--chunk-tokens` or use a shorter source."
        )

    synthesis_dir = work_dir / "synthesis"
    run_self_study_synthesis(
        resource_path=corpus_manifest_path,
        output_dir=synthesis_dir,
        base_url=base_url,
        api_key=api_key,
        run_mode="smoke",
        num_samples=synthesis_num_samples,
        top_logprobs=synthesis_top_logprobs,
        max_context_slices=1,
        max_completion_tokens_a=synthesis_max_completion_tokens_a,
        max_completion_tokens_b=synthesis_max_completion_tokens_b,
    )

    slice_id = corpus_manifest["chunks"][0]["chunk_id"]
    train_dir = work_dir / "train"
    train_summary = train_cartridge(
        dataset_path=synthesis_dir / "conversations.jsonl",
        output_dir=train_dir,
        slice_id=slice_id,
        device=device,
        cartridge_tokens=cartridge_tokens,
        steps=train_steps,
    )

    demo_manifest = {
        "source_path": str(Path(source_path).resolve()),
        "work_dir": str(work_dir.resolve()),
        "corpus_manifest_path": str(corpus_manifest_path.resolve()),
        "synthesis_manifest_path": str((synthesis_dir / "manifest.json").resolve()),
        "slice_id": slice_id,
        "cartridge_path": train_summary["cartridge_path"],
        "checkpoint_path": train_summary["checkpoint_path"],
        "model_id": DEFAULT_MATRIX.model_id,
        "chunk_tokens": chunk_tokens,
        "synthesis_num_samples": synthesis_num_samples,
        "cartridge_tokens": cartridge_tokens,
        "train_steps": train_steps,
    }
    write_json(work_dir / "demo_manifest.json", demo_manifest)
    return demo_manifest


def _load_single_chunk_text(corpus_manifest_path: str | Path) -> tuple[str, str]:
    payload = json.loads(Path(corpus_manifest_path).read_text(encoding="utf-8"))
    chunks = payload["chunks"]
    if len(chunks) != 1:
        raise ValueError("Demo asking path expects exactly one chunk.")
    return chunks[0]["chunk_id"], chunks[0]["text"]


def answer_with_cartridge(
    *,
    cartridge_path: str | Path,
    questions: list[str],
    device: str,
    max_completion_tokens: int = 128,
) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MATRIX.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MATRIX.model_id,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    cartridge = TrainableKVCartridge.load(cartridge_path, device=device)

    answers: list[dict[str, Any]] = []
    eos_token_id = tokenizer.eos_token_id
    with torch.inference_mode():
        for question in questions:
            messages = [
                {"role": "system", "content": CARTRIDGE_SYSTEM_PROMPT},
                {"role": "user", "content": f"/no_think\n{question}"},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
            input_ids = tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"].to(device)

            _sync_if_cuda(device)
            prefill_start = time.perf_counter()
            outputs = model(
                input_ids=input_ids,
                past_key_values=cartridge.as_cache(model.config),
                use_cache=True,
            )
            _sync_if_cuda(device)
            prefill_ms = (time.perf_counter() - prefill_start) * 1000.0

            generated_ids: list[int] = []
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            _sync_if_cuda(device)
            decode_start = time.perf_counter()
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
            decode_ms = (time.perf_counter() - decode_start) * 1000.0

            answers.append(
                {
                    "method": "cartridge",
                    "question": question,
                    "answer": _decode_visible_answer(tokenizer, generated_ids),
                    "prefill_ms": prefill_ms,
                    "decode_tokens_per_second": (
                        len(generated_ids) / (decode_ms / 1000.0)
                        if decode_ms > 0 and generated_ids
                        else None
                    ),
                    "total_latency_ms": prefill_ms + decode_ms,
                    "canonical_kv_bytes": cartridge.canonical_kv_bytes(),
                    "prompt_tokens": int(input_ids.shape[-1]),
                    "completion_tokens": len(generated_ids),
                }
            )
    return answers


def answer_with_full_context(
    *,
    corpus_manifest_path: str | Path,
    questions: list[str],
    device: str,
    max_completion_tokens: int = 128,
) -> list[dict[str, Any]]:
    _, corpus_text = _load_single_chunk_text(corpus_manifest_path)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MATRIX.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MATRIX.model_id,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()

    answers: list[dict[str, Any]] = []
    eos_token_id = tokenizer.eos_token_id
    with torch.inference_mode():
        for question in questions:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.format(context=corpus_text)},
                {"role": "user", "content": f"/no_think\n{question}"},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
            input_ids = tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"].to(device)

            _sync_if_cuda(device)
            prefill_start = time.perf_counter()
            outputs = model(input_ids=input_ids, use_cache=True)
            _sync_if_cuda(device)
            prefill_ms = (time.perf_counter() - prefill_start) * 1000.0

            generated_ids: list[int] = []
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            _sync_if_cuda(device)
            decode_start = time.perf_counter()
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
            decode_ms = (time.perf_counter() - decode_start) * 1000.0

            answers.append(
                {
                    "method": "full_context",
                    "question": question,
                    "answer": _decode_visible_answer(tokenizer, generated_ids),
                    "prefill_ms": prefill_ms,
                    "decode_tokens_per_second": (
                        len(generated_ids) / (decode_ms / 1000.0)
                        if decode_ms > 0 and generated_ids
                        else None
                    ),
                    "total_latency_ms": prefill_ms + decode_ms,
                    "canonical_kv_bytes": canonical_kv_bytes(
                        num_tokens=int(input_ids.shape[-1]),
                        num_hidden_layers=model.config.num_hidden_layers,
                        num_key_value_heads=model.config.num_key_value_heads,
                        head_dim=_head_dim(model.config),
                    ),
                    "prompt_tokens": int(input_ids.shape[-1]),
                    "completion_tokens": len(generated_ids),
                }
            )
    return answers
