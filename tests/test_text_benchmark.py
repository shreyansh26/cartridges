from __future__ import annotations

import json

from cartridges.benchmarks.text_benchmark import (
    BOOTSTRAP_ANSWER_PROMPT,
    _clean_assistant_text,
    _assistant_target_token_ids,
    generate_teacher_answers,
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
