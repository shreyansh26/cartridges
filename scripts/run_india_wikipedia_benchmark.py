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

from cartridges.benchmarks import (  # noqa: E402
    build_eval_rows_from_spec,
    generate_bootstrap_questions,
    write_india_benchmark_report,
)
from cartridges.data.arxiv_smoke import build_arxiv_smoke_manifest  # noqa: E402
from cartridges.data.common import write_json, write_jsonl  # noqa: E402
from cartridges.demo.qa import (  # noqa: E402
    generate_bootstrap_answer_records,
    materialize_bootstrap_rows,
)
from cartridges.eval import run_cartridge_eval, run_local_hf_matched_eval  # noqa: E402
from cartridges.train import train_cartridge  # noqa: E402


def _start_vllm_server(gpu_index: int, port: int, log_path: Path) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "serve_vllm.py"),
        "--port",
        str(port),
        "--max-model-len",
        "12288",
        "--gpu-memory-utilization",
        "0.60",
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


def _load_single_chunk_text(corpus_manifest_path: Path) -> tuple[str, str]:
    payload = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    chunks = payload["chunks"]
    if len(chunks) != 1:
        raise ValueError(f"Expected a single corpus chunk, found {len(chunks)}.")
    return chunks[0]["chunk_id"], chunks[0]["text"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the held-out India Wikipedia benchmark for full-context vs cartridges.",
    )
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--work-dir", default=str(ROOT / "outputs" / "india_wikipedia_benchmark"))
    parser.add_argument("--bootstrap-count", type=int, default=120)
    parser.add_argument("--cartridge-tokens", type=int, default=512)
    parser.add_argument("--train-steps", type=int, default=240)
    parser.add_argument("--max-completion-tokens", type=int, default=48)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = ROOT / "examples" / "india_wikipedia_8192.txt"
    eval_spec_path = ROOT / "examples" / "india_wikipedia_20_eval_spec.json"
    eval_path = work_dir / "india_eval.jsonl"
    bootstrap_questions_path = work_dir / "bootstrap_questions.txt"
    build_dir = work_dir / "cartridge_build"
    corpus_manifest_path = build_dir / "corpus_manifest.json"
    bootstrap_answers_path = build_dir / "bootstrap_answers.jsonl"
    training_dataset_path = build_dir / "train_dataset.jsonl"
    base_url = f"http://127.0.0.1:{args.port}/v1"
    api_key = "cartridges-local"

    build_eval_rows_from_spec(
        corpus_path=corpus_path,
        spec_path=eval_spec_path,
        output_path=eval_path,
    )
    build_arxiv_smoke_manifest(
        source_path=corpus_path,
        output_path=corpus_manifest_path,
        chunk_tokens=8192,
        stride_tokens=8192,
    )
    slice_id, corpus_text = _load_single_chunk_text(corpus_manifest_path)

    build_started = time.perf_counter()
    server = _start_vllm_server(
        gpu_index=args.gpu,
        port=args.port,
        log_path=work_dir / "logs" / "vllm.log",
    )
    try:
        _wait_for_server(base_url=base_url, api_key=api_key)
        bootstrap_questions = generate_bootstrap_questions(
            corpus_path=corpus_path,
            output_path=bootstrap_questions_path,
            base_url=base_url,
            api_key=api_key,
            eval_spec_path=eval_spec_path,
            num_questions=args.bootstrap_count,
        )
        bootstrap_answers = generate_bootstrap_answer_records(
            corpus_text=corpus_text,
            questions=bootstrap_questions,
            base_url=base_url,
            api_key=api_key,
            max_completion_tokens=args.max_completion_tokens,
        )
        write_jsonl(bootstrap_answers_path, bootstrap_answers)
    finally:
        _stop_vllm_server(server, port=args.port)

    training_rows = materialize_bootstrap_rows(
        corpus_text=corpus_text,
        slice_id=slice_id,
        answer_records=bootstrap_answers,
        device=args.device,
        top_logprobs=5,
    )
    write_jsonl(training_dataset_path, training_rows)
    train_summary = train_cartridge(
        dataset_path=training_dataset_path,
        output_dir=build_dir / "train",
        slice_id=slice_id,
        device=args.device,
        cartridge_tokens=args.cartridge_tokens,
        steps=args.train_steps,
    )
    build_seconds = time.perf_counter() - build_started
    manifest = {
        "source_path": str(corpus_path.resolve()),
        "work_dir": str(build_dir.resolve()),
        "corpus_manifest_path": str(corpus_manifest_path.resolve()),
        "training_dataset_path": str(training_dataset_path.resolve()),
        "bootstrap_answers_path": str(bootstrap_answers_path.resolve()),
        "slice_id": slice_id,
        "cartridge_path": train_summary["cartridge_path"],
        "checkpoint_path": train_summary["checkpoint_path"],
        "model_id": train_summary and "Qwen/Qwen3-4B",
        "chunk_tokens": 8192,
        "bootstrap_question_count": len(bootstrap_questions),
        "cartridge_tokens": args.cartridge_tokens,
        "train_steps": args.train_steps,
    }
    write_json(build_dir / "demo_manifest.json", manifest)

    baseline_path = work_dir / "baseline_hf_matched.jsonl"
    cartridge_path = work_dir / "cartridge_hf_matched.jsonl"
    baseline_records = run_local_hf_matched_eval(
        eval_path=eval_path,
        output_path=baseline_path,
        device=args.device,
        max_completion_tokens=args.max_completion_tokens,
    )
    cartridge_records = run_cartridge_eval(
        eval_path=eval_path,
        cartridge_path=manifest["cartridge_path"],
        output_path=cartridge_path,
        device=args.device,
        max_completion_tokens=args.max_completion_tokens,
    )
    report = write_india_benchmark_report(
        baseline_path=baseline_path,
        cartridge_path=cartridge_path,
        output_dir=work_dir / "report",
        build_seconds=build_seconds,
        bootstrap_question_count=len(bootstrap_questions),
        train_steps=args.train_steps,
    )

    summary = {
        "work_dir": str(work_dir.resolve()),
        "manifest_path": str((work_dir / "cartridge_build" / "demo_manifest.json").resolve()),
        "baseline_records": len(baseline_records),
        "cartridge_records": len(cartridge_records),
        "report_path": report["report_path"],
        "summary_path": str((work_dir / "report" / "summary.json").resolve()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
