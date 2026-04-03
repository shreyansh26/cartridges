#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.benchmarks import (  # noqa: E402
    build_training_dataset,
    generate_bootstrap_questions,
    generate_teacher_answers,
    write_budget_report,
    write_run_report,
)
from cartridges.data import (  # noqa: E402
    build_eval_rows_from_spec,
    build_text_manifest,
    load_experiment_inputs,
    load_single_chunk_text,
)
from cartridges.data.common import write_json  # noqa: E402
from cartridges.eval import run_cartridge_eval, run_local_hf_matched_eval  # noqa: E402
from cartridges.train import train_cartridge  # noqa: E402


def _start_vllm_server(
    *,
    gpu_index: int,
    port: int,
    max_model_len: int,
    log_path: Path,
) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "serve_vllm.py"),
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
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


def _copy_inputs(
    *,
    inputs: dict[str, object],
    destination_dir: Path,
) -> dict[str, str]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for key in ("data_path", "eval_spec_path", "metadata_path"):
        source = inputs.get(key)
        if source is None:
            continue
        source_path = Path(str(source))
        target_path = destination_dir / source_path.name
        shutil.copy2(source_path, target_path)
        copied[key] = str(target_path.resolve())
    return copied


def _update_latest_pointer(run_dir: Path, *, output_root: Path, experiment_name: str) -> None:
    latest_path = output_root / experiment_name / "latest"
    if latest_path.is_symlink() or latest_path.exists():
        latest_path.unlink()
    latest_path.symlink_to(Path("runs") / run_dir.name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the generic single-corpus benchmark for full context vs cartridges.",
    )
    parser.add_argument("experiment_name")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    parser.add_argument("--run-name")
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key", default="cartridges-local")
    parser.add_argument("--bootstrap-count", type=int, default=120)
    parser.add_argument("--cartridge-tokens", nargs="+", type=int, default=[1024])
    parser.add_argument("--train-steps", type=int, default=240)
    parser.add_argument("--max-completion-tokens", type=int, default=48)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--semantic-judge", action="store_true")
    parser.add_argument("--judge-device")
    args = parser.parse_args()

    inputs = load_experiment_inputs(args.experiment_name, data_root=args.data_root)
    output_root = Path(args.output_root)
    run_id = args.run_name or datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = output_root / args.experiment_name / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    copied_inputs = _copy_inputs(inputs=inputs, destination_dir=run_dir / "input")
    manifest_path = run_dir / "input" / "corpus_manifest.json"
    eval_rows_path = run_dir / "eval" / "rows.jsonl"
    bootstrap_dir = run_dir / "bootstrap"
    bootstrap_questions_path = bootstrap_dir / "questions.txt"
    teacher_answers_path = bootstrap_dir / "teacher_answers.jsonl"
    training_dataset_path = bootstrap_dir / "train_dataset.jsonl"
    baseline_dir = run_dir / "baseline"
    baseline_predictions_path = baseline_dir / "predictions.jsonl"

    build_text_manifest(
        source_path=inputs["data_path"],
        output_path=manifest_path,
        chunk_tokens=args.max_context_tokens,
        stride_tokens=args.max_context_tokens,
        corpus_id=args.experiment_name,
    )
    slice_id, corpus_text = load_single_chunk_text(manifest_path)
    eval_spec = json.loads(Path(str(inputs["eval_spec_path"])).read_text(encoding="utf-8"))
    build_eval_rows_from_spec(
        corpus_path=inputs["data_path"],
        spec_path=inputs["eval_spec_path"],
        output_path=eval_rows_path,
        sample_id=args.experiment_name,
    )

    managed_server = args.base_url is None
    base_url = args.base_url or f"http://127.0.0.1:{args.port}/v1"
    server = None
    server_max_model_len = args.max_context_tokens + max(args.max_completion_tokens + 256, 512)

    prepare_started = time.perf_counter()
    if managed_server:
        server = _start_vllm_server(
            gpu_index=args.gpu,
            port=args.port,
            max_model_len=server_max_model_len,
            log_path=run_dir / "logs" / "vllm.log",
        )
    try:
        if managed_server:
            _wait_for_server(base_url=base_url, api_key=args.api_key)
        bootstrap_examples = generate_bootstrap_questions(
            corpus_text=corpus_text,
            eval_spec=eval_spec,
            output_path=bootstrap_questions_path,
            base_url=base_url,
            api_key=args.api_key,
            num_questions=args.bootstrap_count,
        )
        teacher_answers = generate_teacher_answers(
            corpus_text=corpus_text,
            bootstrap_examples=bootstrap_examples,
            output_path=teacher_answers_path,
            base_url=base_url,
            api_key=args.api_key,
            max_completion_tokens=args.max_completion_tokens,
        )
    finally:
        if managed_server and server is not None:
            _stop_vllm_server(server, port=args.port)

    build_training_dataset(
        corpus_text=corpus_text,
        slice_id=slice_id,
        answer_records=teacher_answers,
        output_path=training_dataset_path,
        device=args.device,
        top_logprobs=5,
    )
    preparation_seconds = time.perf_counter() - prepare_started

    baseline_records = run_local_hf_matched_eval(
        eval_path=eval_rows_path,
        output_path=baseline_predictions_path,
        device=args.device,
        max_completion_tokens=args.max_completion_tokens,
    )

    budget_summaries: list[dict[str, object]] = []
    for cartridge_tokens in args.cartridge_tokens:
        budget_label = f"cartridge_{cartridge_tokens}"
        budget_dir = run_dir / budget_label
        train_started = time.perf_counter()
        train_summary = train_cartridge(
            dataset_path=training_dataset_path,
            output_dir=budget_dir / "train",
            slice_id=slice_id,
            device=args.device,
            cartridge_tokens=cartridge_tokens,
            steps=args.train_steps,
        )
        train_seconds = time.perf_counter() - train_started
        cartridge_predictions_path = budget_dir / "predictions.jsonl"
        run_cartridge_eval(
            eval_path=eval_rows_path,
            cartridge_path=train_summary["cartridge_path"],
            output_path=cartridge_predictions_path,
            device=args.device,
            max_completion_tokens=args.max_completion_tokens,
        )
        summary = write_budget_report(
            experiment_name=args.experiment_name,
            budget_label=budget_label,
            baseline_path=baseline_predictions_path,
            cartridge_path=cartridge_predictions_path,
            output_dir=budget_dir / "report",
            build_seconds=preparation_seconds + train_seconds,
            bootstrap_question_count=len(bootstrap_examples),
            train_steps=args.train_steps,
            cartridge_tokens=cartridge_tokens,
            semantic_judge=args.semantic_judge,
            judge_device=args.judge_device or args.device,
        )
        write_json(
            budget_dir / "manifest.json",
            {
                "experiment_name": args.experiment_name,
                "cartridge_tokens": cartridge_tokens,
                "train_summary": train_summary,
                "predictions_path": str(cartridge_predictions_path.resolve()),
                "report_path": summary["report_path"],
                "semantic_judge": args.semantic_judge,
            },
        )
        budget_summaries.append(summary)

    aggregate_report = write_run_report(
        experiment_name=args.experiment_name,
        run_dir=run_dir,
        budget_summaries=budget_summaries,
    )
    run_manifest = {
        "experiment_name": args.experiment_name,
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "input_files": copied_inputs,
        "corpus_manifest_path": str(manifest_path.resolve()),
        "eval_rows_path": str(eval_rows_path.resolve()),
        "bootstrap_questions_path": str(bootstrap_questions_path.resolve()),
        "teacher_answers_path": str(teacher_answers_path.resolve()),
        "training_dataset_path": str(training_dataset_path.resolve()),
        "baseline_predictions_path": str(baseline_predictions_path.resolve()),
        "baseline_records": len(baseline_records),
        "budget_reports": [
            {
                "budget_label": item["budget_label"],
                "report_path": item["report_path"],
                "summary_path": str(
                    (run_dir / item["budget_label"] / "report" / "summary.json").resolve()
                ),
            }
            for item in budget_summaries
        ],
        "aggregate_report": aggregate_report,
    }
    write_json(run_dir / "run_manifest.json", run_manifest)
    _update_latest_pointer(run_dir, output_root=output_root, experiment_name=args.experiment_name)

    print(json.dumps(run_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
