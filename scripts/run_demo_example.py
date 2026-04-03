#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.demo import (  # noqa: E402
    answer_with_cartridge,
    answer_with_full_context,
    build_demo_cartridge,
)


def _start_vllm_server(gpu_index: int, port: int, log_path: Path) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "serve_vllm.py"),
        "--port",
        str(port),
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.55",
        "--max-logprobs",
        "8",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=env,
    )


def _wait_for_server(base_url: str, api_key: str, timeout_seconds: float = 240.0) -> None:
    deadline = time.time() + timeout_seconds
    with httpx.Client(timeout=5.0) as client:
        while time.time() < deadline:
            try:
                health = client.get(base_url.replace("/v1", "/health"))
                models = client.get(
                    f"{base_url}/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if health.status_code == 200 and models.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2.0)
    raise TimeoutError(f"Timed out waiting for vLLM server at {base_url}.")


def _stop_vllm_server(process: subprocess.Popen[str], port: int) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=120)
    with httpx.Client(timeout=2.0) as client:
        try:
            client.get(f"http://127.0.0.1:{port}/health")
        except httpx.HTTPError:
            return
    raise RuntimeError(f"vLLM server on port {port} is still reachable after shutdown.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the checked-in single-corpus cartridge demo example.",
    )
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--work-dir", default=str(ROOT / "outputs" / "demo_example"))
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--cartridge-tokens", type=int, default=128)
    parser.add_argument("--max-completion-tokens", type=int, default=64)
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}/v1"
    api_key = "cartridges-local"
    work_dir = Path(args.work_dir)
    question_text = (ROOT / "examples" / "demo_questions.txt").read_text(encoding="utf-8")
    questions = [line.strip() for line in question_text.splitlines() if line.strip()]

    server = _start_vllm_server(
        gpu_index=args.gpu,
        port=args.port,
        log_path=work_dir / "logs" / "vllm.log",
    )
    try:
        _wait_for_server(base_url=base_url, api_key=api_key)
        manifest = build_demo_cartridge(
            source_path=ROOT / "examples" / "demo_corpus.txt",
            work_dir=work_dir,
            base_url=base_url,
            api_key=api_key,
            device=args.device,
            chunk_tokens=2048,
            stride_tokens=2048,
            synthesis_num_samples=0,
            synthesis_top_logprobs=5,
            bootstrap_questions=questions,
            bootstrap_max_completion_tokens=args.max_completion_tokens,
            cartridge_tokens=args.cartridge_tokens,
            train_steps=args.train_steps,
            synthesis_max_completion_tokens_a=16,
            synthesis_max_completion_tokens_b=args.max_completion_tokens,
        )
    finally:
        _stop_vllm_server(server, port=args.port)

    results = {
        "manifest": manifest,
        "cartridge": answer_with_cartridge(
            cartridge_path=manifest["cartridge_path"],
            questions=questions,
            device=args.device,
            max_completion_tokens=args.max_completion_tokens,
        ),
        "full_context": answer_with_full_context(
            corpus_manifest_path=manifest["corpus_manifest_path"],
            questions=questions,
            device=args.device,
            max_completion_tokens=args.max_completion_tokens,
        ),
    }
    output_path = work_dir / "demo_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "work_dir": str(work_dir.resolve()),
                "output_path": str(output_path.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
