#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.demo import (  # noqa: E402
    answer_with_cartridge,
    answer_with_full_context,
    build_demo_cartridge,
)


def _read_questions(args: argparse.Namespace) -> list[str]:
    questions: list[str] = []
    if args.question:
        questions.extend(args.question)
    if args.question_file:
        for line in Path(args.question_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                questions.append(line)
    if not questions:
        raise ValueError("Provide at least one --question or --question-file.")
    return questions


def _write_output(payload: object, output_path: str | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def _run_build(args: argparse.Namespace) -> int:
    manifest = build_demo_cartridge(
        source_path=args.source_path,
        work_dir=args.work_dir,
        base_url=args.base_url,
        api_key=args.api_key,
        device=args.device,
        chunk_tokens=args.chunk_tokens,
        stride_tokens=args.stride_tokens,
        synthesis_num_samples=args.synthesis_num_samples,
        synthesis_top_logprobs=args.synthesis_top_logprobs,
        cartridge_tokens=args.cartridge_tokens,
        train_steps=args.train_steps,
        synthesis_max_completion_tokens_a=args.synthesis_max_completion_tokens_a,
        synthesis_max_completion_tokens_b=args.synthesis_max_completion_tokens_b,
    )
    _write_output(manifest, args.output_path)
    return 0


def _run_ask_once(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest_path).read_text(encoding="utf-8"))
    questions = _read_questions(args)

    results = {
        "cartridge": answer_with_cartridge(
            cartridge_path=manifest["cartridge_path"],
            questions=questions,
            device=args.device,
            max_completion_tokens=args.max_completion_tokens,
        ),
    }
    if args.show_baseline:
        results["full_context"] = answer_with_full_context(
            corpus_manifest_path=manifest["corpus_manifest_path"],
            questions=questions,
            device=args.device,
            max_completion_tokens=args.max_completion_tokens,
        )
    _write_output(results, args.output_path)
    return 0


def _run_interactive(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest_path).read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "mode": "interactive",
                "manifest_path": str(Path(args.manifest_path).resolve()),
                "show_baseline": args.show_baseline,
            },
            indent=2,
            sort_keys=True,
        )
    )
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            print()
            return 0
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            return 0
        payload = {
            "cartridge": answer_with_cartridge(
                cartridge_path=manifest["cartridge_path"],
                questions=[question],
                device=args.device,
                max_completion_tokens=args.max_completion_tokens,
            )[0]
        }
        if args.show_baseline:
            payload["full_context"] = answer_with_full_context(
                corpus_manifest_path=manifest["corpus_manifest_path"],
                questions=[question],
                device=args.device,
                max_completion_tokens=args.max_completion_tokens,
            )[0]
        print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a single-cartridge demo from a long source and query it repeatedly.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build",
        help="Build a demo cartridge from one source file.",
    )
    build_parser.add_argument("--source-path", required=True)
    build_parser.add_argument("--work-dir", required=True)
    build_parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    build_parser.add_argument("--api-key", default="cartridges-local")
    build_parser.add_argument("--device", default="cuda:0")
    build_parser.add_argument("--chunk-tokens", type=int, default=1024)
    build_parser.add_argument("--stride-tokens", type=int, default=None)
    build_parser.add_argument("--synthesis-num-samples", type=int, default=8)
    build_parser.add_argument("--synthesis-top-logprobs", type=int, default=5)
    build_parser.add_argument("--cartridge-tokens", type=int, default=256)
    build_parser.add_argument("--train-steps", type=int, default=30)
    build_parser.add_argument("--synthesis-max-completion-tokens-a", type=int, default=64)
    build_parser.add_argument("--synthesis-max-completion-tokens-b", type=int, default=192)
    build_parser.add_argument("--output-path", default=None)

    ask_parser = subparsers.add_parser("ask", help="Query a previously built demo cartridge.")
    ask_parser.add_argument("--manifest-path", required=True)
    ask_parser.add_argument("--device", default="cuda:0")
    ask_parser.add_argument("--question", action="append", default=[])
    ask_parser.add_argument("--question-file", default=None)
    ask_parser.add_argument("--max-completion-tokens", type=int, default=128)
    ask_parser.add_argument("--show-baseline", action="store_true")
    ask_parser.add_argument("--interactive", action="store_true")
    ask_parser.add_argument("--output-path", default=None)

    args = parser.parse_args()
    if args.command == "build":
        return _run_build(args)
    if args.command == "ask" and args.interactive:
        return _run_interactive(args)
    if args.command == "ask":
        return _run_ask_once(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
