#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.data.arxiv_smoke import build_arxiv_smoke_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the arXiv smoke corpus manifest.")
    parser.add_argument(
        "--source-path",
        default=str(ROOT / "cartridges_ref" / "examples" / "arxiv" / "cartridges.tex"),
    )
    parser.add_argument("--output-path", default=str(ROOT / "data" / "arxiv_smoke" / "manifest.json"))
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--stride-tokens", type=int, default=384)
    args = parser.parse_args()

    manifest = build_arxiv_smoke_manifest(
        source_path=args.source_path,
        output_path=args.output_path,
        chunk_tokens=args.chunk_tokens,
        stride_tokens=args.stride_tokens,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
