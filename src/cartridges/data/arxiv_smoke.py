from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer

from cartridges.config import DEFAULT_MATRIX
from cartridges.data.common import stable_hash


def build_arxiv_smoke_manifest(
    source_path: str | Path,
    output_path: str | Path,
    chunk_tokens: int = 512,
    stride_tokens: int = 384,
) -> dict:
    source_path = Path(source_path)
    output_path = Path(output_path)

    text = source_path.read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MATRIX.model_id)
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    chunks: list[dict] = []
    for start in range(0, max(len(token_ids) - chunk_tokens + 1, 1), stride_tokens):
        window = token_ids[start : start + chunk_tokens]
        if not window:
            break
        chunk_text = tokenizer.decode(window)
        record = {
            "chunk_id": f"arxiv-{start}",
            "start_token": start,
            "end_token": start + len(window),
            "text": chunk_text,
        }
        record["row_hash"] = stable_hash(record)
        chunks.append(record)
        if start + chunk_tokens >= len(token_ids):
            break

    manifest = {
        "corpus_id": "arxiv_smoke",
        "source_path": str(source_path),
        "model_id": DEFAULT_MATRIX.model_id,
        "num_tokens": len(token_ids),
        "num_chunks": len(chunks),
        "chunks": chunks,
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        __import__("json").dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return manifest
