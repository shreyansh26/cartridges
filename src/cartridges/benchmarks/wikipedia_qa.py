from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from cartridges.clients import VLLMClient
from cartridges.data.common import stable_hash, write_json, write_jsonl


def build_eval_rows_from_spec(
    *,
    corpus_path: str | Path,
    spec_path: str | Path,
    output_path: str | Path,
    sample_id: str = "india-wikipedia-8192",
) -> list[dict[str, Any]]:
    context = Path(corpus_path).read_text(encoding="utf-8")
    spec_rows = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item in spec_rows:
        row = {
            "sample_id": sample_id,
            "context": context,
            "query": item["query"],
            "answer_prompt": item["answer_prompt"],
            "answers": item["answers"],
            "question_id": item["id"],
        }
        row["row_hash"] = stable_hash(
            {
                "sample_id": row["sample_id"],
                "query": row["query"],
                "answer_prompt": row["answer_prompt"],
                "answers": row["answers"],
                "question_id": row["question_id"],
            }
        )
        rows.append(row)
    write_jsonl(Path(output_path), rows)
    return rows


def _parse_question_answer_lines(text: str) -> list[tuple[str, str]]:
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    if text.startswith("<think>"):
        text = text.removeprefix("<think>").strip()
    pairs: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+[\).\s-]+", "", line)
        line = re.sub(r"^[-*]\s+", "", line)
        line = line.strip()
        if "|||" not in line:
            continue
        question, answer = [part.strip() for part in line.split("|||", maxsplit=1)]
        if not question.endswith("?"):
            continue
        if not answer:
            continue
        pairs.append((question, answer))
    return pairs


def _content_passages(corpus_text: str) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in corpus_text.split("\n\n") if paragraph.strip()]
    content_paragraphs = [
        paragraph
        for paragraph in paragraphs
        if not re.fullmatch(r"=+\s*[^=]+?\s*=+", paragraph)
    ]
    sentence_chunks: list[str] = []
    for paragraph in content_paragraphs:
        if len(paragraph) <= 1800:
            sentence_chunks.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        buffer: list[str] = []
        buffer_chars = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            projected = buffer_chars + len(sentence) + (1 if buffer else 0)
            if buffer and projected > 1500:
                sentence_chunks.append(" ".join(buffer))
                buffer = [sentence]
                buffer_chars = len(sentence)
            else:
                buffer.append(sentence)
                buffer_chars = projected
        if buffer:
            sentence_chunks.append(" ".join(buffer))

    passages: list[str] = []
    buffer: list[str] = []
    buffer_chars = 0
    for chunk in sentence_chunks:
        projected = buffer_chars + len(chunk) + (2 if buffer else 0)
        if buffer and projected > 1800:
            passages.append("\n\n".join(buffer))
            buffer = [chunk]
            buffer_chars = len(chunk)
        else:
            buffer.append(chunk)
            buffer_chars = projected
    if buffer:
        passages.append("\n\n".join(buffer))
    return passages


def generate_bootstrap_questions(
    *,
    corpus_path: str | Path,
    output_path: str | Path,
    base_url: str,
    api_key: str,
    eval_spec_path: str | Path,
    num_questions: int,
    batch_size: int = 20,
    max_rounds: int = 40,
) -> list[str]:
    corpus_text = Path(corpus_path).read_text(encoding="utf-8")
    eval_spec = json.loads(Path(eval_spec_path).read_text(encoding="utf-8"))
    forbidden = {item["query"].strip().lower() for item in eval_spec}
    client = VLLMClient(base_url=base_url, api_key=api_key)
    generated: list[str] = []
    seen = set(forbidden)
    passages = _content_passages(corpus_text)
    if not passages:
        raise RuntimeError("Bootstrap question generation found no content passages.")
    system_prompt = (
        "You generate factual question prompts for training a retrieval model. "
        "Do not emit <think> tags. Every question must be answerable directly from the provided "
        "corpus, must have a short exact answer copied from the corpus, and must not require "
        "external knowledge."
    )
    try:
        for round_index in range(max_rounds):
            if len(generated) >= num_questions:
                break
            for passage_index, passage in enumerate(passages):
                if len(generated) >= num_questions:
                    break
                remaining = num_questions - len(generated)
                prompt_batch_size = min(batch_size, max(4, remaining))
                recent_seen = sorted(seen)[-20:]
                user_prompt = (
                    "/no_think\nPassage:\n"
                    f"{passage}\n\n"
                    f"Round {round_index + 1}, passage {passage_index + 1} of {len(passages)}.\n"
                    f"Generate {prompt_batch_size} distinct factual training examples from this "
                    "passage only. Each line must have the format QUESTION ||| ANSWER. "
                    "The answer must be a short exact substring copied from this passage. "
                    "Do not ask any question whose answer is not explicitly stated in the passage. "
                    "Avoid any question already in this list:\n"
                    + "\n".join(recent_seen)
                    + "\n\nReturn only lines in that format with no numbering and no extra text."
                )
                result = client.chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_completion_tokens=512,
                    temperature=0.9,
                    run_mode="smoke",
                )
                passage_lower = passage.lower()
                for question, answer in _parse_question_answer_lines(result.text):
                    if answer.lower() not in passage_lower:
                        continue
                    lowered = question.lower()
                    if lowered in seen:
                        continue
                    seen.add(lowered)
                    generated.append(question)
                    if len(generated) >= num_questions:
                        break
    finally:
        client.close()
    minimum_questions = min(num_questions, max(20, num_questions // 2))
    if len(generated) < minimum_questions:
        raise RuntimeError(
            f"Generated only {len(generated)} bootstrap questions; minimum required is "
            f"{minimum_questions}."
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(generated) + "\n", encoding="utf-8")
    return generated


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def write_india_benchmark_report(
    *,
    baseline_path: str | Path,
    cartridge_path: str | Path,
    output_dir: str | Path,
    build_seconds: float,
    bootstrap_question_count: int,
    train_steps: int,
) -> dict[str, Any]:
    baseline_rows = [
        json.loads(line)
        for line in Path(baseline_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cartridge_rows = [
        json.loads(line)
        for line in Path(cartridge_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(baseline_rows) != len(cartridge_rows):
        raise ValueError("Baseline and cartridge row counts differ.")

    paired_rows: list[dict[str, Any]] = []
    for baseline, cartridge in zip(baseline_rows, cartridge_rows, strict=True):
        paired_rows.append(
            {
                "prompt_id": baseline["prompt_id"],
                "question": baseline["metadata"]["question_id"],
                "baseline_exact_match": baseline["exact_match"],
                "cartridge_exact_match": cartridge["exact_match"],
                "baseline_prediction": baseline["prediction"],
                "cartridge_prediction": cartridge["prediction"],
                "gold": baseline["gold"],
                "baseline_prefill_ms": baseline.get("prefill_ms"),
                "cartridge_prefill_ms": cartridge.get("prefill_ms"),
                "baseline_total_latency_ms": baseline.get("total_latency_ms"),
                "cartridge_total_latency_ms": cartridge.get("total_latency_ms"),
                "baseline_decode_tokens_per_second": baseline.get("decode_tokens_per_second"),
                "cartridge_decode_tokens_per_second": cartridge.get("decode_tokens_per_second"),
                "compression_ratio": baseline["canonical_kv_bytes"]
                / cartridge["canonical_kv_bytes"],
                "throughput_ratio": (
                    cartridge["decode_tokens_per_second"] / baseline["decode_tokens_per_second"]
                    if baseline.get("decode_tokens_per_second")
                    and cartridge.get("decode_tokens_per_second")
                    else None
                ),
                "prefill_speedup_ratio": (
                    baseline["prefill_ms"] / cartridge["prefill_ms"]
                    if baseline.get("prefill_ms") and cartridge.get("prefill_ms")
                    else None
                ),
                "end_to_end_speedup_ratio": (
                    baseline["total_latency_ms"] / cartridge["total_latency_ms"]
                    if baseline.get("total_latency_ms") and cartridge.get("total_latency_ms")
                    else None
                ),
            }
        )

    baseline_first = paired_rows[0]
    followups = paired_rows[1:]
    summary = {
        "num_questions": len(paired_rows),
        "bootstrap_question_count": bootstrap_question_count,
        "train_steps": train_steps,
        "compression_build_seconds": build_seconds,
        "baseline_exact_match_rate": sum(int(row["baseline_exact_match"]) for row in paired_rows)
        / len(paired_rows),
        "cartridge_exact_match_rate": sum(
            int(row["cartridge_exact_match"]) for row in paired_rows
        )
        / len(paired_rows),
        "avg_compression_ratio": _safe_mean([row["compression_ratio"] for row in paired_rows]),
        "avg_throughput_ratio": _safe_mean(
            [
                row["throughput_ratio"]
                for row in paired_rows
                if row["throughput_ratio"] is not None
            ]
        ),
        "avg_prefill_speedup_ratio": _safe_mean(
            [
                row["prefill_speedup_ratio"]
                for row in paired_rows
                if row["prefill_speedup_ratio"] is not None
            ]
        ),
        "avg_end_to_end_speedup_ratio": _safe_mean(
            [
                row["end_to_end_speedup_ratio"]
                for row in paired_rows
                if row["end_to_end_speedup_ratio"] is not None
            ]
        ),
        "baseline_first_total_latency_ms": baseline_first["baseline_total_latency_ms"],
        "cartridge_first_total_latency_ms": baseline_first["cartridge_total_latency_ms"],
        "baseline_followup_total_latency_ms": _safe_mean(
            [row["baseline_total_latency_ms"] for row in followups]
        ),
        "cartridge_followup_total_latency_ms": _safe_mean(
            [row["cartridge_total_latency_ms"] for row in followups]
        ),
        "baseline_first_prefill_ms": baseline_first["baseline_prefill_ms"],
        "cartridge_first_prefill_ms": baseline_first["cartridge_prefill_ms"],
        "baseline_followup_prefill_ms": _safe_mean(
            [row["baseline_prefill_ms"] for row in followups]
        ),
        "cartridge_followup_prefill_ms": _safe_mean(
            [row["cartridge_prefill_ms"] for row in followups]
        ),
        "baseline_session_total_ms": sum(
            row["baseline_total_latency_ms"]
            for row in paired_rows
            if row["baseline_total_latency_ms"]
        ),
        "cartridge_query_session_total_ms": sum(
            row["cartridge_total_latency_ms"]
            for row in paired_rows
            if row["cartridge_total_latency_ms"]
        ),
    }
    summary["cartridge_amortized_session_total_ms"] = (
        summary["cartridge_query_session_total_ms"] + (build_seconds * 1000.0)
    )
    followup_delta = None
    if (
        summary["baseline_followup_total_latency_ms"] is not None
        and summary["cartridge_followup_total_latency_ms"] is not None
    ):
        followup_delta = (
            summary["baseline_followup_total_latency_ms"]
            - summary["cartridge_followup_total_latency_ms"]
        )
    summary["followup_latency_advantage_ms"] = followup_delta
    summary["break_even_query_count"] = None
    if followup_delta and followup_delta > 0:
        summary["break_even_query_count"] = (build_seconds * 1000.0) / followup_delta

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "comparison.jsonl", paired_rows)

    lines = [
        "# India Wikipedia Cartridge Benchmark",
        "",
        f"- Questions: {summary['num_questions']}",
        f"- Bootstrap questions: {bootstrap_question_count}",
        f"- Train steps: {train_steps}",
        f"- One-time compression/build time: {build_seconds:.2f}s",
        f"- Baseline exact-match rate: {summary['baseline_exact_match_rate']:.3f}",
        f"- Cartridge exact-match rate: {summary['cartridge_exact_match_rate']:.3f}",
        f"- Average compression ratio: {summary['avg_compression_ratio']:.3f}x",
        (
            f"- Average throughput ratio: {summary['avg_throughput_ratio']:.3f}x"
            if summary["avg_throughput_ratio"] is not None
            else "- Average throughput ratio: n/a"
        ),
        (
            f"- Average prefill speedup: {summary['avg_prefill_speedup_ratio']:.3f}x"
            if summary["avg_prefill_speedup_ratio"] is not None
            else "- Average prefill speedup: n/a"
        ),
        (
            f"- Average end-to-end speedup: {summary['avg_end_to_end_speedup_ratio']:.3f}x"
            if summary["avg_end_to_end_speedup_ratio"] is not None
            else "- Average end-to-end speedup: n/a"
        ),
        f"- Baseline first-query latency: {summary['baseline_first_total_latency_ms']:.2f} ms",
        f"- Cartridge first-query latency: {summary['cartridge_first_total_latency_ms']:.2f} ms",
        (
            f"- Baseline follow-up mean latency: "
            f"{summary['baseline_followup_total_latency_ms']:.2f} ms"
            if summary["baseline_followup_total_latency_ms"] is not None
            else "- Baseline follow-up mean latency: n/a"
        ),
        (
            f"- Cartridge follow-up mean latency: "
            f"{summary['cartridge_followup_total_latency_ms']:.2f} ms"
            if summary["cartridge_followup_total_latency_ms"] is not None
            else "- Cartridge follow-up mean latency: n/a"
        ),
        f"- Baseline session total latency: {summary['baseline_session_total_ms']:.2f} ms",
        (
            f"- Cartridge query-only session latency: "
            f"{summary['cartridge_query_session_total_ms']:.2f} ms"
        ),
        (
            f"- Cartridge amortized session latency: "
            f"{summary['cartridge_amortized_session_total_ms']:.2f} ms"
        ),
        (
            f"- Break-even query count: {summary['break_even_query_count']:.2f}"
            if summary["break_even_query_count"] is not None
            else "- Break-even query count: n/a"
        ),
        "",
        (
            "| question | baseline_em | cartridge_em | compression_ratio | "
            "throughput_ratio | prefill_speedup | e2e_speedup |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in paired_rows:
        throughput = row["throughput_ratio"]
        prefill = row["prefill_speedup_ratio"]
        e2e = row["end_to_end_speedup_ratio"]
        lines.append(
            f"| {row['question']} | {int(row['baseline_exact_match'])} | "
            f"{int(row['cartridge_exact_match'])} | {row['compression_ratio']:.3f} | "
            f"{throughput:.3f} | {prefill:.3f} | {e2e:.3f} |"
            if throughput is not None and prefill is not None and e2e is not None
            else
            f"| {row['question']} | {int(row['baseline_exact_match'])} | "
            f"{int(row['cartridge_exact_match'])} | {row['compression_ratio']:.3f} | "
            "n/a | n/a | n/a |"
        )
    report_path = output_dir / "comparison.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        **summary,
        "comparison_path": str((output_dir / "comparison.jsonl").resolve()),
        "report_path": str(report_path.resolve()),
    }
