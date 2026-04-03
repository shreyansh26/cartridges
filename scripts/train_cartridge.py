#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.train import train_cartridge  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a cartridge on synthesized conversations.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--slice-id", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cartridge-tokens", type=int, default=256)
    parser.add_argument("--num-frozen-tokens", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()

    summary = train_cartridge(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        slice_id=args.slice_id,
        device=args.device,
        cartridge_tokens=args.cartridge_tokens,
        num_frozen_tokens=args.num_frozen_tokens,
        learning_rate=args.learning_rate,
        steps=args.steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        resume_from=args.resume_from,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
