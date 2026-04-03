from __future__ import annotations

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
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No evaluation rows found in {path}.")
    return rows


def write_eval_records(path: str | Path, records: list[EvalRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    user_prompt = f"/no_think\n{row['query']}\n\n{row['answer_prompt']}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=row["context"])},
        {"role": "user", "content": user_prompt},
    ]


def build_cartridge_messages(row: dict[str, Any]) -> list[dict[str, str]]:
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
    return (
        num_tokens
        * num_hidden_layers
        * num_key_value_heads
        * head_dim
        * 2
        * dtype_bytes
    )


def normalize_prediction(text: str) -> list[str]:
    cleaned = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    cleaned = cleaned.replace("<think>", " ").replace("</think>", " ")
    number_matches = re.findall(r"\d+", cleaned)
    if number_matches:
        return sorted(number_matches)
    return [re.sub(r"\s+", " ", cleaned).strip().lower()]


def exact_match(prediction: str, gold: list[str]) -> bool:
    return normalize_prediction(prediction) == sorted(str(item) for item in gold)
