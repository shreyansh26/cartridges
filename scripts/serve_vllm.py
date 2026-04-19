#!/usr/bin/env python3
"""Launch a foreground vLLM server with repo defaults."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.config import DEFAULT_MATRIX, DEFAULT_VLLM_PORT  # noqa: E402


def main() -> int:
    """Build the ``vllm serve`` command line and proxy its exit code."""
    parser = argparse.ArgumentParser(
        description="Launch a foreground vLLM OpenAI-compatible server."
    )
    parser.add_argument("--model", default=DEFAULT_MATRIX.model_id)
    parser.add_argument("--revision", default=os.environ.get("CARTRIDGES_MODEL_REVISION"))
    parser.add_argument("--tokenizer", default=os.environ.get("CARTRIDGES_TOKENIZER_ID"))
    parser.add_argument(
        "--tokenizer-revision",
        default=os.environ.get(
            "CARTRIDGES_TOKENIZER_REVISION",
            os.environ.get("CARTRIDGES_MODEL_REVISION"),
        ),
    )
    parser.add_argument(
        "--served-model-name",
        default=os.environ.get("CARTRIDGES_SERVED_MODEL_NAME"),
    )
    parser.add_argument("--port", type=int, default=DEFAULT_VLLM_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-logprobs", type=int, default=20)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "cartridges-local"))
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--disable-tokenizer-info-endpoint", action="store_true")
    args = parser.parse_args()

    command = [
        "vllm",
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--api-key",
        args.api_key,
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--seed",
        str(args.seed),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-logprobs",
        str(args.max_logprobs),
    ]
    if args.revision:
        command.extend(["--revision", args.revision])
    if args.tokenizer:
        command.extend(["--tokenizer", args.tokenizer])
    if args.tokenizer_revision:
        command.extend(["--tokenizer-revision", args.tokenizer_revision])
    if args.served_model_name:
        command.extend(["--served-model-name", args.served_model_name])
    if args.enable_prefix_caching:
        command.append("--enable-prefix-caching")
    if not args.disable_tokenizer_info_endpoint:
        command.append("--enable-tokenizer-info-endpoint")

    # Keep this wrapper intentionally thin so the benchmark runner and manual usage share defaults.
    print("Launching:", " ".join(command))
    process = subprocess.run(command, check=False)
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
