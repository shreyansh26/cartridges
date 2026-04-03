#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.eval import run_local_hf_matched_eval, run_vllm_quality_eval  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run baseline NIAH evaluation.")
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="cartridges-local")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=128)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("vllm_quality", "hf_matched"),
        default=("vllm_quality", "hf_matched"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {}
    if "vllm_quality" in args.methods:
        records = run_vllm_quality_eval(
            eval_path=args.eval_path,
            output_path=output_dir / "baseline_vllm_quality.jsonl",
            base_url=args.base_url,
            api_key=args.api_key,
            max_samples=args.max_samples,
            max_completion_tokens=args.max_completion_tokens,
        )
        results["vllm_quality_records"] = len(records)

    if "hf_matched" in args.methods:
        records = run_local_hf_matched_eval(
            eval_path=args.eval_path,
            output_path=output_dir / "baseline_hf_matched.jsonl",
            device=args.device,
            max_samples=args.max_samples,
            max_completion_tokens=args.max_completion_tokens,
        )
        results["hf_matched_records"] = len(records)

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
