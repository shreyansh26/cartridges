"""Load and materialize the repo's standardized ``data/<experiment>/`` layout."""

import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from cartridges.config import DEFAULT_MATRIX
from cartridges.data.common import stable_hash, write_json, write_jsonl


def resolve_experiment_dir(
    experiment_name: str,
    *,
    data_root: str | Path,
) -> Path:
    """Resolve ``data/<experiment_name>`` and fail if it does not exist."""
    experiment_dir = Path(data_root) / experiment_name
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    return experiment_dir


def load_experiment_inputs(
    experiment_name: str,
    *,
    data_root: str | Path,
) -> dict[str, Any]:
    """Load the required input files and optional metadata for one experiment."""
    experiment_dir = resolve_experiment_dir(experiment_name, data_root=data_root)
    data_path = experiment_dir / "data.txt"
    eval_spec_path = experiment_dir / "eval_spec.json"
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing corpus file: {data_path}")
    if not eval_spec_path.is_file():
        raise FileNotFoundError(f"Missing eval spec: {eval_spec_path}")

    metadata_path = experiment_dir / "metadata.json"
    metadata = None
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return {
        "experiment_name": experiment_name,
        "experiment_dir": experiment_dir,
        "data_path": data_path,
        "eval_spec_path": eval_spec_path,
        "metadata_path": metadata_path if metadata_path.is_file() else None,
        "metadata": metadata,
    }


def build_text_manifest(
    *,
    source_path: str | Path,
    output_path: str | Path,
    chunk_tokens: int,
    stride_tokens: int | None = None,
    corpus_id: str | None = None,
) -> dict[str, Any]:
    """Tokenize ``data.txt`` into one or more fixed-size windows and write a manifest."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    stride_tokens = stride_tokens or chunk_tokens

    text = source_path.read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MATRIX.model_id)
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    chunks: list[dict[str, Any]] = []
    if len(token_ids) <= chunk_tokens:
        start_offsets = [0]
    else:
        final_start = len(token_ids) - chunk_tokens
        start_offsets = list(range(0, final_start + 1, stride_tokens))
        if start_offsets[-1] != final_start:
            start_offsets.append(final_start)

    for start in start_offsets:
        window = token_ids[start : start + chunk_tokens]
        if not window:
            break
        record = {
            "chunk_id": f"text-{start}",
            "start_token": start,
            "end_token": start + len(window),
            "text": tokenizer.decode(window),
        }
        record["row_hash"] = stable_hash(record)
        chunks.append(record)

    manifest = {
        "corpus_id": corpus_id or source_path.parent.name,
        "source_path": str(source_path.resolve()),
        "model_id": DEFAULT_MATRIX.model_id,
        "num_tokens": len(token_ids),
        "num_chunks": len(chunks),
        "chunks": chunks,
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    write_json(output_path, manifest)
    return manifest


def load_corpus_slices(manifest_path: str | Path) -> list[dict[str, Any]]:
    """Load all chunk records from a corpus manifest in a stable order."""
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    chunks = payload["chunks"]
    if not chunks:
        raise ValueError(f"No chunks found in {manifest_path}.")
    return [
        {
            "chunk_id": chunk["chunk_id"],
            "start_token": int(chunk["start_token"]),
            "end_token": int(chunk["end_token"]),
            "text": chunk["text"],
            "row_hash": chunk["row_hash"],
        }
        for chunk in chunks
    ]


def build_eval_rows_from_spec(
    *,
    corpus_path: str | Path,
    spec_path: str | Path,
    output_path: str | Path,
    sample_id: str,
) -> list[dict[str, Any]]:
    """Expand ``eval_spec.json`` into JSONL rows that the evaluators consume directly."""
    context = Path(corpus_path).read_text(encoding="utf-8")
    spec_rows = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item in spec_rows:
        row = {
            "sample_id": sample_id,
            "context": context,
            "query": item["query"],
            "answer_prompt": item["answer_prompt"],
            "answers": item["answers"],
            "question_id": item["id"],
        }
        row["row_hash"] = stable_hash(
            {
                "sample_id": row["sample_id"],
                "query": row["query"],
                "answer_prompt": row["answer_prompt"],
                "answers": row["answers"],
                "question_id": row["question_id"],
            }
        )
        rows.append(row)
    write_jsonl(Path(output_path), rows)
    return rows
