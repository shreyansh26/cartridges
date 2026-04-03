#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.data.niah import build_niah_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a deterministic local NIAH benchmark.")
    parser.add_argument(
        "--source-path",
        default=str(ROOT / "cartridges_ref" / "examples" / "arxiv" / "cartridges.tex"),
    )
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "niah"))
    parser.add_argument("--max-seq-length", type=int, default=32768)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--num-needle-keys", type=int, default=2)
    parser.add_argument("--values-per-key", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = build_niah_dataset(
        source_path=args.source_path,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        num_samples=args.num_samples,
        num_needle_keys=args.num_needle_keys,
        values_per_key=args.values_per_key,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
