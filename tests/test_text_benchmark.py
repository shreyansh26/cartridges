from __future__ import annotations

import json
from unittest.mock import patch

from cartridges.benchmarks.text_benchmark import (
    BOOTSTRAP_ANSWER_PROMPT,
    _clean_assistant_text,
    _assistant_target_token_ids,
    SemanticEquivalenceJudge,
    generate_teacher_answers,
    write_budget_report,
    write_run_report,
)


def test_clean_assistant_text_strips_think_tags() -> None:
    raw = "<think>\nreasoning\n</think>\n Republic of India \n"
    assert _clean_assistant_text(raw) == "Republic of India"


def test_generate_teacher_answers_preserves_exact_bootstrap_answers(tmp_path) -> None:
    output_path = tmp_path / "teacher_answers.jsonl"
    answer_records = generate_teacher_answers(
        corpus_text="unused",
        bootstrap_examples=[
            {
                "question": "What is India's official name?",
                "expected_answer": "<think>hidden</think> Republic of India",
            }
        ],
        output_path=output_path,
        base_url="http://unused",
        api_key="unused",
        max_completion_tokens=48,
    )

    assert answer_records == [
        {
            "question": "What is India's official name?",
            "user_message": (
                "/no_think\nWhat is India's official name?\n\n"
                f"{BOOTSTRAP_ANSWER_PROMPT}"
            ),
            "assistant_text": "Republic of India",
            "expected_answer": "Republic of India",
        }
    ]
    stored_rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    assert stored_rows == answer_records


class _FakeTokenizer:
    eos_token_id = 42

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [7, 8, 9] if text == "Republic of India" else []


def test_assistant_target_token_ids_appends_eos() -> None:
    tokenizer = _FakeTokenizer()
    assert _assistant_target_token_ids(tokenizer, "Republic of India") == [7, 8, 9, 42]


def test_write_budget_report_with_semantic_judge(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    cartridge_path = tmp_path / "cartridge.jsonl"
    baseline_row = {
        "prompt_id": "p1",
        "prediction": "The Bay of Bengal.",
        "gold": ["bay of bengal"],
        "exact_match": False,
        "canonical_kv_bytes": 100,
        "compression_ratio": 1.0,
        "prefill_ms": 10.0,
        "decode_tokens_per_second": 20.0,
        "total_latency_ms": 30.0,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "metadata": {
            "question_id": "q1",
            "query": "Which body of water lies to the southeast of India?",
        },
    }
    cartridge_row = {
        **baseline_row,
        "prediction": "Bay of Bengal",
        "canonical_kv_bytes": 10,
        "prefill_ms": 2.0,
        "decode_tokens_per_second": 25.0,
        "total_latency_ms": 8.0,
    }
    baseline_path.write_text(json.dumps(baseline_row) + "\n", encoding="utf-8")
    cartridge_path.write_text(json.dumps(cartridge_row) + "\n", encoding="utf-8")

    class _FakeJudge:
        def __init__(self, *, device: str, model_id: str = "") -> None:
            self.device = device
            self.model_id = model_id

        def is_equivalent(self, *, question: str, references: list[str], candidate: str) -> bool:
            assert question == "Which body of water lies to the southeast of India?"
            assert references == ["bay of bengal"]
            return "bay of bengal" in candidate.lower()

        def close(self) -> None:
            return None

    with patch(
        "cartridges.benchmarks.text_benchmark.SemanticEquivalenceJudge",
        _FakeJudge,
    ):
        summary = write_budget_report(
            experiment_name="demo",
            budget_label="cartridge_128",
            baseline_path=baseline_path,
            cartridge_path=cartridge_path,
            output_dir=tmp_path / "report",
            build_seconds=1.0,
            bootstrap_question_count=4,
            train_steps=8,
            cartridge_tokens=128,
            semantic_judge=True,
            judge_device="cpu",
        )

    assert summary["baseline_semantic_match_rate"] == 1.0
    assert summary["cartridge_semantic_match_rate"] == 1.0
    comparison_rows = [
        json.loads(line)
        for line in (tmp_path / "report" / "comparison.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert comparison_rows[0]["baseline_semantic_match"] is True
    assert comparison_rows[0]["cartridge_semantic_match"] is True


def test_write_run_report_surfaces_semantic_columns(tmp_path) -> None:
    summary = write_run_report(
        experiment_name="demo",
        run_dir=tmp_path,
        budget_summaries=[
            {
                "budget_label": "cartridge_128",
                "baseline_exact_match_rate": 0.9,
                "cartridge_exact_match_rate": 0.8,
                "baseline_semantic_match_rate": 1.0,
                "cartridge_semantic_match_rate": 0.95,
                "avg_compression_ratio": 16.0,
                "avg_throughput_ratio": 1.1,
                "avg_prefill_speedup_ratio": 8.0,
                "avg_end_to_end_speedup_ratio": 1.9,
                "compression_build_seconds": 12.0,
                "baseline_followup_total_latency_ms": 300.0,
                "cartridge_followup_total_latency_ms": 180.0,
            }
        ],
    )

    assert summary["semantic_judge_enabled"] is True
    report_path = tmp_path / "report" / "comparison.md"
    report_text = report_path.read_text(encoding="utf-8")
    assert "baseline_sem" in report_text
    assert "cartridge_sem" in report_text
    assert "| cartridge_128 | 0.900 | 0.800 | 1.000 | 0.950 |" in report_text


def test_semantic_judge_heuristic_handles_wrappers() -> None:
    assert SemanticEquivalenceJudge._heuristic_equivalent(
        references=["Republic of India"],
        candidate="The Republic of India.",
    )
    assert SemanticEquivalenceJudge._heuristic_equivalent(
        references=["Bay of Bengal"],
        candidate="Bay of Bengal.",
    )
    assert not SemanticEquivalenceJudge._heuristic_equivalent(
        references=["Buddhism and Jainism"],
        candidate="Hinduism and Buddhism",
    )
