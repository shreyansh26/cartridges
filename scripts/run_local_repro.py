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
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.data.common import write_json  # noqa: E402
from cartridges.data.niah import build_niah_dataset  # noqa: E402
from cartridges.eval import (  # noqa: E402
    merge_results,
    run_cartridge_eval,
    run_local_hf_matched_eval,
    run_vllm_quality_eval,
)
from cartridges.synthesis import run_self_study_synthesis  # noqa: E402
from cartridges.train import train_cartridge  # noqa: E402


def _load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _run_env_check(mode: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_env.py"), "--mode", mode, "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _write_sentinel(run_dir: Path, phase: str, payload: dict) -> None:
    write_json(run_dir / ".done" / f"{phase}.json", payload)


def _load_sentinel(run_dir: Path, phase: str) -> dict | None:
    path = run_dir / ".done" / f"{phase}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _start_vllm_server(config: dict, gpu_index: int, log_path: Path) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    command = [
        "vllm",
        "serve",
        "Qwen/Qwen3-4B",
        "--host",
        config["host"],
        "--port",
        str(config["port"]),
        "--api-key",
        config["api_key"],
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--seed",
        "42",
        "--max-model-len",
        str(config["max_model_len"]),
        "--gpu-memory-utilization",
        "0.85",
        "--max-logprobs",
        "20",
        "--enable-tokenizer-info-endpoint",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        start_new_session=True,
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


def _first_slice_id(conversations_path: Path) -> str:
    first_line = conversations_path.read_text(encoding="utf-8").splitlines()[0]
    row = json.loads(first_line)
    return row["slice_ids"][0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the local single-GPU cartridges smoke/full flow."
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else ROOT / "configs" / f"{args.mode}.yaml"
    config = _load_yaml(config_path)
    env_payload = _run_env_check(args.mode)
    gpu_index = int(env_payload["selected_gpu"])

    if args.resume_from:
        run_dir = Path(args.resume_from)
    else:
        run_name = args.run_name or f"{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir = ROOT / "outputs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "env_check.json", env_payload)

    data_dir = run_dir / "data" / "niah"
    synthesis_dir = run_dir / "synthesis"
    eval_dir = run_dir / "eval"
    train_dir = run_dir / "train"
    report_dir = run_dir / "report"

    if _load_sentinel(run_dir, "prepare_niah") is None:
        manifest = build_niah_dataset(
            source_path=ROOT / config["source_path"],
            output_dir=data_dir,
            max_seq_length=int(config["max_seq_length"]),
            num_samples=int(config["niah_num_samples"]),
        )
        _write_sentinel(run_dir, "prepare_niah", manifest)

    base_url = f"http://{config['vllm']['host']}:{config['vllm']['port']}/v1"
    vllm_process = None
    try:
        if (
            _load_sentinel(run_dir, "synthesis") is None
            or _load_sentinel(run_dir, "baseline_vllm") is None
        ):
            vllm_process = _start_vllm_server(
                config["vllm"],
                gpu_index=gpu_index,
                log_path=run_dir / "logs" / "vllm.log",
            )
            _wait_for_server(base_url=base_url, api_key=config["vllm"]["api_key"])

            if _load_sentinel(run_dir, "synthesis") is None:
                synthesis_manifest = run_self_study_synthesis(
                    resource_path=data_dir / "samples.json",
                    output_dir=synthesis_dir,
                    base_url=base_url,
                    api_key=config["vllm"]["api_key"],
                    run_mode=config["mode"],
                    num_samples=int(config["synthesis_num_samples"]),
                    top_logprobs=int(config["synthesis_top_logprobs"]),
                    max_context_slices=int(config["synthesis_max_context_slices"]),
                    max_completion_tokens_a=int(
                        config.get("synthesis_max_completion_tokens_a", 128)
                    ),
                    max_completion_tokens_b=int(
                        config.get("synthesis_max_completion_tokens_b", 384)
                    ),
                )
                _write_sentinel(run_dir, "synthesis", synthesis_manifest)

            if _load_sentinel(run_dir, "baseline_vllm") is None:
                records = run_vllm_quality_eval(
                    eval_path=data_dir / "eval.jsonl",
                    output_path=eval_dir / "baseline_vllm_quality.jsonl",
                    base_url=base_url,
                    api_key=config["vllm"]["api_key"],
                    max_samples=config["baseline_max_samples"],
                    max_completion_tokens=int(config["max_completion_tokens"]),
                )
                _write_sentinel(run_dir, "baseline_vllm", {"records": len(records)})
    finally:
        if vllm_process is not None:
            _stop_vllm_server(vllm_process, port=int(config["vllm"]["port"]))

    conversations_path = synthesis_dir / "conversations.jsonl"
    slice_id = _first_slice_id(conversations_path)

    if _load_sentinel(run_dir, "train") is None:
        summary = train_cartridge(
            dataset_path=conversations_path,
            output_dir=train_dir,
            slice_id=slice_id,
            device=f"cuda:{gpu_index}",
            cartridge_tokens=int(config["cartridge_tokens"]),
            steps=int(config["train_steps"]),
        )
        _write_sentinel(run_dir, "train", summary)

    if _load_sentinel(run_dir, "baseline_hf") is None:
        records = run_local_hf_matched_eval(
            eval_path=data_dir / "eval.jsonl",
            output_path=eval_dir / "baseline_hf_matched.jsonl",
            device=f"cuda:{gpu_index}",
            max_samples=config["baseline_max_samples"],
            max_completion_tokens=int(config["max_completion_tokens"]),
        )
        _write_sentinel(run_dir, "baseline_hf", {"records": len(records)})

    if _load_sentinel(run_dir, "cartridge_eval") is None:
        records = run_cartridge_eval(
            eval_path=data_dir / "eval.jsonl",
            cartridge_path=train_dir / f"{slice_id}_cartridge.pt",
            output_path=eval_dir / "cartridge_eval.jsonl",
            device=f"cuda:{gpu_index}",
            sample_id=slice_id,
            max_samples=config["cartridge_eval_max_samples"],
            max_completion_tokens=int(config["max_completion_tokens"]),
        )
        _write_sentinel(run_dir, "cartridge_eval", {"records": len(records)})

    if _load_sentinel(run_dir, "report") is None:
        summary = merge_results(
            baseline_path=eval_dir / "baseline_hf_matched.jsonl",
            cartridge_path=eval_dir / "cartridge_eval.jsonl",
            output_dir=report_dir,
        )
        _write_sentinel(run_dir, "report", summary)

    print(json.dumps({"run_dir": str(run_dir.resolve()), "slice_id": slice_id}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
