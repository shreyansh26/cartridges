#!/usr/bin/env python3
"""Run the end-to-end benchmark for one dataset directory under ``data/``."""

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
    build_retrieval_index,
    generate_bootstrap_questions,
    generate_teacher_answers,
    route_eval_questions,
    write_budget_report,
    write_run_report,
)
from cartridges.data import (  # noqa: E402
    build_eval_rows_from_spec,
    build_text_manifest,
    load_corpus_slices,
    load_experiment_inputs,
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
    """Launch the local vLLM teacher server used during bootstrap synthesis."""
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
    """Block until the managed vLLM server responds to health and model probes."""
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


def _gpu_uuid_for_index(gpu_index: int) -> str:
    """Resolve a physical GPU index to the UUID used by ``nvidia-smi`` process queries."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        index_text, uuid = [item.strip() for item in line.split(",", maxsplit=1)]
        if int(index_text) == gpu_index:
            return uuid
    raise RuntimeError(f"Could not resolve GPU UUID for index {gpu_index}.")


def _kill_lingering_vllm_gpu_workers(gpu_index: int) -> None:
    """Kill orphaned vLLM worker processes that still hold memory on the managed GPU."""
    gpu_uuid = _gpu_uuid_for_index(gpu_index)
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    candidate_pids: list[int] = []
    for line in result.stdout.splitlines():
        gpu_uuid_text, pid_text, process_name = [item.strip() for item in line.split(",", maxsplit=2)]
        if gpu_uuid_text != gpu_uuid:
            continue
        if "vllm" not in process_name.lower():
            continue
        candidate_pids.append(int(pid_text))
    for pid in candidate_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    deadline = time.time() + 10.0
    while time.time() < deadline:
        alive = []
        for pid in candidate_pids:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            alive.append(pid)
        if not alive:
            return
        time.sleep(0.5)
    for pid in candidate_pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue


def _stop_vllm_server(process: subprocess.Popen[str], port: int, gpu_index: int) -> None:
    """Stop the managed vLLM process, clear leaked workers, and verify the port is gone."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=120)
    _kill_lingering_vllm_gpu_workers(gpu_index)
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
    """Copy dataset inputs into the run directory for reproducible outputs."""
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
    """Point ``outputs/<experiment>/latest`` at the run that just completed."""
    latest_path = output_root / experiment_name / "latest"
    if latest_path.is_symlink() or latest_path.exists():
        latest_path.unlink()
    latest_path.symlink_to(Path("runs") / run_dir.name)


def main() -> int:
    """Execute the full benchmark pipeline for one experiment directory."""
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
    parser.add_argument("--train-learning-rate", type=float, default=3e-3)
    parser.add_argument("--train-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--train-validation-examples", type=int, default=16)
    parser.add_argument("--train-validation-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-completion-tokens", type=int, default=48)
    parser.add_argument("--chunk-tokens", type=int, default=8192)
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
    retrieval_dir = run_dir / "retrieval"
    baseline_dir = run_dir / "baseline"
    baseline_predictions_path = baseline_dir / "predictions.jsonl"

    # Materialize the dataset into a manifest plus exact-match eval rows before any model work starts.
    stride_tokens = max(1, int(args.chunk_tokens * 0.85))
    build_text_manifest(
        source_path=inputs["data_path"],
        output_path=manifest_path,
        chunk_tokens=args.chunk_tokens,
        stride_tokens=stride_tokens,
        corpus_id=args.experiment_name,
    )
    slices = load_corpus_slices(manifest_path)
    eval_spec = json.loads(Path(str(inputs["eval_spec_path"])).read_text(encoding="utf-8"))
    eval_rows = build_eval_rows_from_spec(
        corpus_path=inputs["data_path"],
        spec_path=inputs["eval_spec_path"],
        output_path=eval_rows_path,
        sample_id=args.experiment_name,
    )
    retrieval_index = build_retrieval_index(slices=slices, output_dir=retrieval_dir)
    retrieval_routes = route_eval_questions(
        eval_rows=eval_rows,
        slices=slices,
        retrieval_dir=retrieval_dir,
    )

    managed_server = args.base_url is None
    base_url = args.base_url or f"http://127.0.0.1:{args.port}/v1"
    server = None
    server_max_model_len = args.chunk_tokens + max(args.max_completion_tokens + 256, 512)

    prepare_started = time.perf_counter()
    bootstrap_artifacts: list[dict[str, object]] = []
    teacher_answers_by_slice: dict[str, list[dict[str, str]]] = {}
    training_dataset_paths: dict[str, str] = {}
    bootstrap_question_total = 0
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
        # Generate teacher questions and exact answers independently for each chunk.
        for slice_record in slices:
            slice_id = str(slice_record["chunk_id"])
            slice_dir = bootstrap_dir / slice_id
            bootstrap_questions_path = slice_dir / "questions.txt"
            teacher_answers_path = slice_dir / "teacher_answers.jsonl"
            bootstrap_examples = generate_bootstrap_questions(
                corpus_text=str(slice_record["text"]),
                eval_spec=eval_spec,
                output_path=bootstrap_questions_path,
                base_url=base_url,
                api_key=args.api_key,
                num_questions=args.bootstrap_count,
            )
            teacher_answers = generate_teacher_answers(
                corpus_text=str(slice_record["text"]),
                bootstrap_examples=bootstrap_examples,
                output_path=teacher_answers_path,
                base_url=base_url,
                api_key=args.api_key,
                max_completion_tokens=args.max_completion_tokens,
            )
            teacher_answers_by_slice[slice_id] = teacher_answers
            bootstrap_question_total += len(bootstrap_examples)
            bootstrap_artifacts.append(
                {
                    "slice_id": slice_id,
                    "questions_path": str(bootstrap_questions_path.resolve()),
                    "teacher_answers_path": str(teacher_answers_path.resolve()),
                    "bootstrap_question_count": len(bootstrap_examples),
                }
            )
    finally:
        if managed_server and server is not None:
            _stop_vllm_server(server, port=args.port, gpu_index=args.gpu)

    # Convert teacher answers into token-level supervision after vLLM is gone so the local
    # HF teacher can reuse the GPU safely on single-card machines.
    for slice_record in slices:
        slice_id = str(slice_record["chunk_id"])
        training_dataset_path = bootstrap_dir / slice_id / "train_dataset.jsonl"
        build_training_dataset(
            corpus_text=str(slice_record["text"]),
            slice_id=slice_id,
            answer_records=teacher_answers_by_slice[slice_id],
            output_path=training_dataset_path,
            device=args.device,
            top_logprobs=5,
        )
        training_dataset_paths[slice_id] = str(training_dataset_path.resolve())
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
        slice_train_summaries: dict[str, dict[str, object]] = {}
        cartridge_paths: dict[str, str] = {}
        train_seconds = 0.0
        for slice_record in slices:
            slice_id = str(slice_record["chunk_id"])
            train_started = time.perf_counter()
            # Each slice trains its own compact cache for the requested budget.
            train_summary = train_cartridge(
                dataset_path=training_dataset_paths[slice_id],
                output_dir=budget_dir / "train" / slice_id,
                slice_id=slice_id,
                device=args.device,
                cartridge_tokens=cartridge_tokens,
                learning_rate=args.train_learning_rate,
                steps=args.train_steps,
                max_grad_norm=args.train_max_grad_norm,
                seed=args.seed,
                validation_examples=args.train_validation_examples,
                validation_interval=args.train_validation_interval,
            )
            train_seconds += time.perf_counter() - train_started
            slice_train_summaries[slice_id] = train_summary
            cartridge_paths[slice_id] = str(train_summary["cartridge_path"])
        cartridge_predictions_path = budget_dir / "predictions.jsonl"
        # Cartridge inference uses the same HF backend as the matched baseline, with top-1
        # chunk routing selecting which slice-specific cache to inject for each question.
        run_cartridge_eval(
            eval_path=eval_rows_path,
            cartridge_paths=cartridge_paths,
            retrieval_routes=retrieval_routes,
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
            bootstrap_question_count=bootstrap_question_total,
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
                "slice_train_summaries": slice_train_summaries,
                "cartridge_paths": cartridge_paths,
                "predictions_path": str(cartridge_predictions_path.resolve()),
                "report_path": summary["report_path"],
                "semantic_judge": args.semantic_judge,
            },
        )
        budget_summaries.append(summary)

    # The run manifest is the stable index for everything generated during one invocation.
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
        "retrieval_index_path": str((retrieval_dir / "index.json").resolve()),
        "retrieval_routes_path": str((retrieval_dir / "routes.jsonl").resolve()),
        "retrieval_summary_path": str((retrieval_dir / "summary.json").resolve()),
        "slice_count": len(slices),
        "chunk_tokens": args.chunk_tokens,
        "stride_tokens": stride_tokens,
        "bootstrap_slices": bootstrap_artifacts,
        "training_datasets": training_dataset_paths,
        "baseline_predictions_path": str(baseline_predictions_path.resolve()),
        "baseline_records": len(baseline_records),
        "retrieval_index": retrieval_index,
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
