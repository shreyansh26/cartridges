"""Small serialization helpers shared across data preparation and reporting."""

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def canonical_json(data: Any) -> str:
    """Encode data with stable key ordering so hashes and JSONL rows are reproducible."""
    if is_dataclass(data):
        data = asdict(data)
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(data: Any) -> str:
    """Return a SHA-256 hash over the canonical JSON form of ``data``."""
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def write_json(path: Path, data: Any) -> None:
    """Write pretty JSON and create parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write canonical JSONL rows so artifacts diff cleanly across runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")
