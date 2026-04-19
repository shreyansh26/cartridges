#!/usr/bin/env python3
"""Validate local environment prerequisites for smoke and full benchmark runs."""

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.config import (  # noqa: E402
    DEFAULT_MATRIX,
    DISALLOWED_GPUS,
    GPU_USED_MEMORY_LIMIT_MIB,
    GPU_UTILIZATION_LIMIT,
    PREFERRED_GPU_ORDER,
    REQUIRED_PORTS,
    resolve_wandb_mode,
)


@dataclass
class GPUInfo:
    """Snapshot of one GPU's availability and active compute processes."""
    index: int
    uuid: str
    memory_total_mib: int
    memory_used_mib: int
    utilization_gpu: int
    active_processes: list[dict[str, str]]


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and capture stdout/stderr without raising automatically."""
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _query_gpus() -> list[GPUInfo]:
    """Read GPU inventory plus active compute processes from ``nvidia-smi``."""
    gpu_result = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if gpu_result.returncode != 0:
        raise RuntimeError(gpu_result.stderr.strip() or "Failed to query GPUs with nvidia-smi.")

    proc_result = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )

    by_uuid: dict[str, list[dict[str, str]]] = {}
    if proc_result.returncode == 0 and proc_result.stdout.strip():
        proc_rows = csv.reader(proc_result.stdout.strip().splitlines())
        for row in proc_rows:
            if len(row) != 4:
                continue
            uuid, pid, process_name, used_memory = [item.strip() for item in row]
            by_uuid.setdefault(uuid, []).append(
                {
                    "pid": pid,
                    "process_name": process_name,
                    "used_gpu_memory_mib": used_memory,
                }
            )

    gpus: list[GPUInfo] = []
    for row in csv.reader(gpu_result.stdout.strip().splitlines()):
        if len(row) != 5:
            continue
        index, uuid, mem_total, mem_used, util = [item.strip() for item in row]
        gpus.append(
            GPUInfo(
                index=int(index),
                uuid=uuid,
                memory_total_mib=int(mem_total),
                memory_used_mib=int(mem_used),
                utilization_gpu=int(util),
                active_processes=by_uuid.get(uuid, []),
            )
        )
    return gpus


def _port_is_free(port: int) -> bool:
    """Return whether the local vLLM port can be bound on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _select_gpu(mode: str, gpus: list[GPUInfo]) -> tuple[GPUInfo | None, list[str]]:
    """Pick the first allowed GPU that satisfies the repo's selection policy."""
    errors: list[str] = []
    allowed = [gpu for gpu in gpus if gpu.index not in DISALLOWED_GPUS]
    allowed_by_index = {gpu.index: gpu for gpu in allowed}

    for index in PREFERRED_GPU_ORDER:
        gpu = allowed_by_index.get(index)
        if gpu is None:
            errors.append(f"Preferred GPU {index} is not present.")
            continue
        if gpu.active_processes:
            errors.append(f"GPU {index} has active compute processes.")
            continue
        if gpu.memory_used_mib > GPU_USED_MEMORY_LIMIT_MIB[mode]:
            errors.append(
                f"GPU {index} has {gpu.memory_used_mib} MiB in use; limit is {GPU_USED_MEMORY_LIMIT_MIB[mode]} MiB."
            )
            continue
        if gpu.utilization_gpu > GPU_UTILIZATION_LIMIT[mode]:
            errors.append(
                f"GPU {index} has {gpu.utilization_gpu}% utilization; limit is {GPU_UTILIZATION_LIMIT[mode]}%."
            )
            continue
        return gpu, errors

    return None, errors


def main() -> int:
    """Validate ports, auth, W&B mode, and GPU availability for one run mode."""
    parser = argparse.ArgumentParser(description="Validate local environment prerequisites.")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    port = args.port if args.port is not None else REQUIRED_PORTS[args.mode]
    warnings: list[str] = []
    errors: list[str] = []

    try:
        wandb_mode, wandb_warnings = resolve_wandb_mode(args.mode)
        warnings.extend(wandb_warnings)
    except RuntimeError as exc:
        wandb_mode = "blocked"
        errors.append(str(exc))

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        errors.append("HF_TOKEN or HUGGINGFACE_HUB_TOKEN must be set.")

    if not _port_is_free(port):
        errors.append(f"Port {port} is already bound.")

    try:
        gpus = _query_gpus()
    except RuntimeError as exc:
        errors.append(str(exc))
        gpus = []

    selected_gpu, gpu_errors = _select_gpu(args.mode, gpus) if gpus else (None, [])
    if selected_gpu is None:
        errors.extend(gpu_errors or ["No allowed GPU satisfied the selection policy."])

    payload = {
        "mode": args.mode,
        "compatibility_matrix": asdict(DEFAULT_MATRIX),
        "port": port,
        "wandb_mode": wandb_mode,
        "warnings": warnings,
        "errors": errors,
        "selected_gpu": selected_gpu.index if selected_gpu is not None else None,
        "gpus": [asdict(gpu) for gpu in gpus],
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"mode={args.mode}")
        print(f"wandb_mode={wandb_mode}")
        print(f"port={port}")
        print(f"compatibility_matrix={asdict(DEFAULT_MATRIX)}")
        if selected_gpu is not None:
            print(f"selected_gpu={selected_gpu.index}")
        for warning in warnings:
            print(f"warning={warning}")
        for error in errors:
            print(f"error={error}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
