#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.config import DEFAULT_VLLM_PORT  # noqa: E402
from cartridges.synthesis import run_self_study_synthesis  # noqa: E402


def _default_resource_path(mode: str) -> Path:
    if mode == "full":
        return ROOT / "data" / "niah_run1" / "samples.json"
    return ROOT / "data" / "arxiv_smoke" / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SELF-STUDY synthesis against a vLLM server.")
    parser.add_argument("--resource-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--base-url", default=f"http://127.0.0.1:{DEFAULT_VLLM_PORT}/v1")
    parser.add_argument("--api-key", default="cartridges-local")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--max-context-slices", type=int, default=2)
    parser.add_argument("--teacher-device", default=None)
    args = parser.parse_args()

    resource_path = Path(args.resource_path) if args.resource_path else _default_resource_path(args.mode)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / "outputs" / f"self_study_{args.mode}"
    )
    num_samples = args.num_samples if args.num_samples is not None else (8 if args.mode == "smoke" else 4096)

    manifest = run_self_study_synthesis(
        resource_path=resource_path,
        output_dir=output_dir,
        base_url=args.base_url,
        api_key=args.api_key,
        run_mode=args.mode,
        num_samples=num_samples,
        seed=args.seed,
        top_logprobs=args.top_logprobs,
        max_context_slices=args.max_context_slices,
        teacher_device=args.teacher_device,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
