from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from cartridges.config import DEFAULT_MATRIX
from cartridges.data.common import stable_hash, write_json, write_jsonl

WORDS = [
    "amber", "azure", "cinder", "delta", "ember", "flux", "glacier", "harbor",
    "ion", "jasmine", "kepler", "lattice", "mango", "nebula", "onyx", "prairie",
    "quartz", "ripple", "saffron", "tundra", "umbra", "velvet", "willow", "xenon",
    "yonder", "zephyr",
]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _random_key(rng: random.Random) -> str:
    return f"{rng.choice(WORDS)}-{rng.choice(WORDS)}"


def _random_value(rng: random.Random) -> str:
    return f"{rng.randint(10_000, 99_999)}"


def _split_evenly(tokens: list[int], parts: int) -> list[list[int]]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    return [
        tokens[round(i * len(tokens) / parts) : round((i + 1) * len(tokens) / parts)]
        for i in range(parts)
    ]


def build_niah_dataset(
    source_path: str | Path,
    output_dir: str | Path,
    max_seq_length: int = 32768,
    num_samples: int = 4,
    num_needle_keys: int = 2,
    values_per_key: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MATRIX.model_id)
    base_text = _normalize_text(source_path.read_text(encoding="utf-8"))
    base_tokens = tokenizer.encode(base_text, add_special_tokens=False)
    if not base_tokens:
        raise ValueError(f"Source corpus at {source_path} produced no tokens.")

    samples: list[dict[str, Any]] = []
    for sample_idx in range(num_samples):
        rng = random.Random(seed + sample_idx)
        expected: dict[str, list[str]] = defaultdict(list)
        insertion_token_blocks: list[list[int]] = []
        for _ in range(num_needle_keys * values_per_key):
            key = _random_key(rng)
            value = _random_value(rng)
            expected[key].append(value)
            sentence = f"One of the special magic numbers for {key} is: {value}."
            insertion_token_blocks.append(
                tokenizer.encode(f"{sentence}\n", add_special_tokens=False)
            )

        queries = []
        for key, answers in expected.items():
            query = {
                "query": f"What are all the special magic numbers for {key} mentioned in the provided text?",
                "answers": sorted(answers),
                "answer_prompt": (
                    f'Please answer with the following format. '
                    f'"The special magic numbers for {key} mentioned in the provided text are: '
                    f'{{ {", ".join(sorted(answers))} }}"'
                ),
            }
            query["row_hash"] = stable_hash(query)
            queries.append(query)

        sample_context = _build_context_with_budget(
            tokenizer=tokenizer,
            base_tokens=base_tokens,
            insertion_token_blocks=insertion_token_blocks,
            queries=queries,
            max_seq_length=max_seq_length,
        )

        sample = {
            "sample_id": f"niah-{sample_idx}",
            "context": sample_context,
            "queries": queries,
            "expected": {key: sorted(values) for key, values in sorted(expected.items())},
        }
        sample["row_hash"] = stable_hash(sample)
        samples.append(sample)

    eval_rows = []
    for sample in samples:
        for query in sample["queries"]:
            prompt_tokens = tokenizer.encode(
                f"{sample['context']}\n\n{query['query']}\n\n{query['answer_prompt']}",
                add_special_tokens=False,
            )
            if len(prompt_tokens) > max_seq_length:
                raise ValueError(
                    f"Prompt for {sample['sample_id']} exceeded max_seq_length={max_seq_length} with {len(prompt_tokens)} tokens."
                )
            row = {
                "sample_id": sample["sample_id"],
                "context": sample["context"],
                **query,
            }
            row["row_hash"] = stable_hash(row)
            eval_rows.append(row)

    manifest = {
        "corpus_id": "niah_local",
        "source_path": str(source_path),
        "model_id": DEFAULT_MATRIX.model_id,
        "max_seq_length": max_seq_length,
        "num_samples": num_samples,
        "num_needle_keys": num_needle_keys,
        "values_per_key": values_per_key,
        "seed": seed,
        "samples_path": "samples.json",
        "eval_path": "eval.jsonl",
        "sample_hashes": [sample["row_hash"] for sample in samples],
        "eval_hashes": [row["row_hash"] for row in eval_rows],
    }
    manifest["manifest_hash"] = stable_hash(manifest)

    write_json(output_dir / "samples.json", {"samples": samples})
    write_jsonl(output_dir / "eval.jsonl", eval_rows)
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _build_context_with_budget(
    tokenizer: AutoTokenizer,
    base_tokens: list[int],
    insertion_token_blocks: list[list[int]],
    queries: list[dict[str, Any]],
    max_seq_length: int,
) -> str:
    def prompt_len(candidate_context: str, query: dict[str, Any]) -> int:
        return len(
            tokenizer.encode(
                f"{candidate_context}\n\n{query['query']}\n\n{query['answer_prompt']}",
                add_special_tokens=False,
            )
        )

    max_query_overhead = max(prompt_len("", query) for query in queries)
    insertion_budget = sum(len(block) for block in insertion_token_blocks)
    reserved_budget = max_query_overhead + insertion_budget + 32
    if reserved_budget >= max_seq_length:
        raise ValueError(
            f"Impossible NIAH budget: reserved {reserved_budget} tokens for query and needles "
            f"with max_seq_length={max_seq_length}."
        )

    trim_margin = 32
    while True:
        context_budget = max_seq_length - max_query_overhead - insertion_budget - trim_margin
        if context_budget <= 0:
            raise ValueError(
                f"Context budget became non-positive with max_seq_length={max_seq_length}."
            )

        repeated_tokens = (base_tokens * math.ceil(context_budget / len(base_tokens)))[:context_budget]
        chunks = _split_evenly(repeated_tokens, len(insertion_token_blocks) + 1)
        context_tokens: list[int] = []
        for idx, chunk in enumerate(chunks):
            context_tokens.extend(chunk)
            if idx < len(insertion_token_blocks):
                context_tokens.extend(insertion_token_blocks[idx])

        candidate_context = tokenizer.decode(context_tokens)
        max_prompt_tokens = max(prompt_len(candidate_context, query) for query in queries)
        if max_prompt_tokens <= max_seq_length:
            return candidate_context

        trim_margin += max_prompt_tokens - max_seq_length + 16
