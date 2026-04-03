#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.eval import run_cartridge_eval  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cartridge-backed NIAH evaluation.")
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--cartridge-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=128)
    args = parser.parse_args()

    records = run_cartridge_eval(
        eval_path=args.eval_path,
        cartridge_path=args.cartridge_path,
        output_path=args.output_path,
        device=args.device,
        sample_id=args.sample_id,
        max_samples=args.max_samples,
        max_completion_tokens=args.max_completion_tokens,
    )
    print(json.dumps({"records": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
