from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cartridges.data.common import write_json, write_jsonl


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def merge_results(
    *,
    baseline_path: str | Path,
    cartridge_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    baseline_rows = _load_jsonl(baseline_path)
    cartridge_rows = _load_jsonl(cartridge_path)
    baseline_by_prompt = {row["prompt_id"]: row for row in baseline_rows}
    cartridge_by_prompt = {row["prompt_id"]: row for row in cartridge_rows}

    shared_prompt_ids = sorted(set(baseline_by_prompt) & set(cartridge_by_prompt))
    if not shared_prompt_ids:
        raise ValueError("No shared prompt ids between baseline and cartridge results.")

    merged_rows: list[dict[str, Any]] = []
    for prompt_id in shared_prompt_ids:
        baseline = baseline_by_prompt[prompt_id]
        cartridge = cartridge_by_prompt[prompt_id]
        baseline_decode = baseline.get("decode_tokens_per_second")
        cartridge_decode = cartridge.get("decode_tokens_per_second")
        throughput_ratio = None
        if baseline_decode and cartridge_decode:
            throughput_ratio = cartridge_decode / baseline_decode
        merged_rows.append(
            {
                "prompt_id": prompt_id,
                "baseline_method": baseline["method"],
                "cartridge_method": cartridge["method"],
                "baseline_exact_match": baseline["exact_match"],
                "cartridge_exact_match": cartridge["exact_match"],
                "baseline_canonical_kv_bytes": baseline["canonical_kv_bytes"],
                "cartridge_canonical_kv_bytes": cartridge["canonical_kv_bytes"],
                "compression_ratio": (
                    baseline["canonical_kv_bytes"] / cartridge["canonical_kv_bytes"]
                ),
                "baseline_decode_tokens_per_second": baseline_decode,
                "cartridge_decode_tokens_per_second": cartridge_decode,
                "throughput_ratio": throughput_ratio,
                "baseline_total_latency_ms": baseline.get("total_latency_ms"),
                "cartridge_total_latency_ms": cartridge.get("total_latency_ms"),
                "baseline_prediction": baseline["prediction"],
                "cartridge_prediction": cartridge["prediction"],
                "gold": baseline["gold"],
            }
        )

    avg_compression_ratio = sum(row["compression_ratio"] for row in merged_rows) / len(merged_rows)
    throughput_rows = [
        row["throughput_ratio"]
        for row in merged_rows
        if row["throughput_ratio"] is not None
    ]
    avg_throughput_ratio = None
    if throughput_rows:
        avg_throughput_ratio = sum(throughput_rows) / len(throughput_rows)

    baseline_accuracy = (
        sum(int(row["baseline_exact_match"]) for row in merged_rows) / len(merged_rows)
    )
    cartridge_accuracy = (
        sum(int(row["cartridge_exact_match"]) for row in merged_rows) / len(merged_rows)
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "comparison.jsonl", merged_rows)

    summary = {
        "baseline_path": str(Path(baseline_path).resolve()),
        "cartridge_path": str(Path(cartridge_path).resolve()),
        "num_pairs": len(merged_rows),
        "avg_compression_ratio": avg_compression_ratio,
        "avg_throughput_ratio": avg_throughput_ratio,
        "baseline_exact_match_rate": baseline_accuracy,
        "cartridge_exact_match_rate": cartridge_accuracy,
    }
    write_json(output_dir / "summary.json", summary)

    markdown_lines = [
        "# Cartridge Smoke Comparison",
        "",
        f"- Paired prompts: {len(merged_rows)}",
        f"- Baseline exact-match rate: {baseline_accuracy:.3f}",
        f"- Cartridge exact-match rate: {cartridge_accuracy:.3f}",
        f"- Average compression ratio: {avg_compression_ratio:.3f}x",
        (
            f"- Average throughput ratio: {avg_throughput_ratio:.3f}x"
            if avg_throughput_ratio is not None
            else "- Average throughput ratio: n/a"
        ),
        "",
        "| prompt_id | baseline_em | cartridge_em | compression_ratio | throughput_ratio |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in merged_rows:
        throughput_value = row["throughput_ratio"]
        throughput_display = f"{throughput_value:.3f}" if throughput_value is not None else "n/a"
        markdown_lines.append(
            f"| {row['prompt_id']} | {int(row['baseline_exact_match'])} | "
            f"{int(row['cartridge_exact_match'])} | {row['compression_ratio']:.3f} | "
            f"{throughput_display} |"
        )
    report_path = output_dir / "comparison.md"
    report_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    return {
        **summary,
        "comparison_path": str((output_dir / "comparison.jsonl").resolve()),
        "report_path": str(report_path.resolve()),
    }
