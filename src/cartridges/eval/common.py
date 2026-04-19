from __future__ import annotations

"""Shared prompt building and scoring helpers for baseline and cartridge evaluation."""

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

SYSTEM_PROMPT = """Please answer the user's question using only the provided context.

<context>
{context}
</context>

Follow the requested answer format exactly. Do not emit <think> tags or chain-of-thought."""

class EvalRecord(BaseModel):
    """Normalized evaluation record written by both baseline and cartridge evaluators."""
    model_config = ConfigDict(extra="forbid")

    prompt_id: str
    method: str
    prediction: str
    gold: list[str]
    exact_match: bool
    canonical_kv_bytes: int
    compression_ratio: float
    prefill_ms: float | None = None
    decode_tokens_per_second: float | None = None
    total_latency_ms: float | None = None
    prompt_tokens: int
    completion_tokens: int
    metadata: dict[str, Any]


def load_eval_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load evaluation rows from JSONL and fail loudly on empty inputs."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No evaluation rows found in {path}.")
    return rows


def write_eval_records(path: str | Path, records: list[EvalRecord]) -> None:
    """Serialize evaluation records as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Build the full-context baseline prompt with the corpus embedded as a system message."""
    user_prompt = f"/no_think\n{row['query']}\n\n{row['answer_prompt']}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=row["context"])},
        {"role": "user", "content": user_prompt},
    ]


def build_cartridge_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Build the cartridge prompt that assumes context already lives inside the KV cache."""
    user_prompt = f"/no_think\n{row['query']}\n\n{row['answer_prompt']}"
    return [{"role": "user", "content": user_prompt}]


def canonical_kv_bytes(
    *,
    num_tokens: int,
    num_hidden_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> int:
    """Compute the exact KV-cache footprint implied by token and model dimensions."""
    return (
        num_tokens
        * num_hidden_layers
        * num_key_value_heads
        * head_dim
        * 2
        * dtype_bytes
    )


def normalize_prediction(text: str) -> list[str]:
    """Normalize free-form predictions for the repo's strict exact-match metric."""
    cleaned = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    cleaned = cleaned.replace("<think>", " ").replace("</think>", " ")
    number_matches = re.findall(r"\d+", cleaned)
    if number_matches:
        return sorted(number_matches)
    return [re.sub(r"\s+", " ", cleaned).strip().lower()]


def exact_match(prediction: str, gold: list[str]) -> bool:
    """Apply the benchmark's exact-match rule after normalization."""
    return normalize_prediction(prediction) == sorted(str(item) for item in gold)
