from __future__ import annotations

"""Benchmark helpers for bootstrap synthesis, supervision building, and reporting."""

import json
import re
import string
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cartridges.clients import VLLMClient
from cartridges.config import DEFAULT_MATRIX
from cartridges.data.common import stable_hash, write_json, write_jsonl

JUDGE_SYSTEM_PROMPT = """You are a strict answer-equivalence judge.

Determine whether the candidate answer means the same thing as any reference answer for the given question.
Use only the question and answer texts provided. Do not use outside world knowledge, corpus knowledge, or unstated assumptions.
Ignore harmless differences in articles, casing, punctuation, or short explanatory wrappers when the same answer is clearly present.
If the candidate changes the meaning, gives a different entity or value, adds contradictory information, or is not clearly the same answer, choose DIFFERENT.

Respond with exactly SAME or DIFFERENT."""


def _parse_question_answer_lines(text: str) -> list[tuple[str, str]]:
    """Parse ``QUESTION ||| ANSWER`` rows out of model output."""
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
        if "|||" not in line:
            continue
        question, answer = [part.strip() for part in line.split("|||", maxsplit=1)]
        if question.endswith("?") and answer:
            pairs.append((question, answer))
    return pairs


def _content_passages(corpus_text: str) -> list[str]:
    """Split corpus text into medium-sized passages for bootstrap question generation."""
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
    buffer = []
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


BOOTSTRAP_ANSWER_PROMPT = (
    "Answer with only the shortest exact phrase copied from the context "
    "that fully answers the question."
)


def _clean_assistant_text(text: str) -> str:
    """Remove reasoning tags and normalize whitespace in teacher/model output."""
    cleaned = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    if cleaned.startswith("<think>"):
        cleaned = cleaned.removeprefix("<think>").strip()
    return re.sub(r"\s+", " ", cleaned).strip()


def _assistant_target_token_ids(tokenizer, assistant_text: str) -> list[int]:
    """Encode the supervised answer and append EOS so the cartridge learns when to stop."""
    assistant_token_ids = tokenizer.encode(assistant_text, add_special_tokens=False)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must expose an eos_token_id for cartridge training.")
    return assistant_token_ids + [int(eos_token_id)]


def generate_bootstrap_questions(
    *,
    corpus_text: str,
    eval_spec: list[dict[str, Any]],
    output_path: str | Path,
    base_url: str,
    api_key: str,
    num_questions: int,
    batch_size: int = 20,
    max_rounds: int = 40,
) -> list[dict[str, str]]:
    """Use the teacher model to synthesize short factual Q/A pairs from the corpus."""
    forbidden = {item["query"].strip().lower() for item in eval_spec}
    client = VLLMClient(base_url=base_url, api_key=api_key)
    generated: list[dict[str, str]] = []
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
                # Feed a local passage instead of the full document so bootstrap questions stay
                # grounded, cheap to generate, and easy to answer with copied text spans.
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
                    generated.append(
                        {
                            "question": question,
                            "expected_answer": answer,
                        }
                    )
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
    output_path.write_text(
        "\n".join(item["question"] for item in generated) + "\n",
        encoding="utf-8",
    )
    return generated


def generate_teacher_answers(
    *,
    corpus_text: str,
    bootstrap_examples: list[dict[str, str]],
    output_path: str | Path,
    base_url: str,
    api_key: str,
    max_completion_tokens: int,
) -> list[dict[str, str]]:
    """Materialize the exact teacher answers that later become cartridge supervision."""
    del corpus_text
    del base_url
    del api_key
    del max_completion_tokens

    if not bootstrap_examples:
        return []

    answer_records: list[dict[str, str]] = []
    for example in bootstrap_examples:
        cleaned_question = example["question"].strip()
        expected_answer = _clean_assistant_text(example["expected_answer"])
        if not cleaned_question or not expected_answer:
            continue
        answer_records.append(
            {
                "question": cleaned_question,
                "user_message": f"/no_think\n{cleaned_question}\n\n{BOOTSTRAP_ANSWER_PROMPT}",
                "assistant_text": expected_answer,
                "expected_answer": expected_answer,
            }
        )
    write_jsonl(Path(output_path), answer_records)
    return answer_records


def build_training_dataset(
    *,
    corpus_text: str,
    slice_id: str,
    answer_records: list[dict[str, str]],
    output_path: str | Path,
    device: str,
    top_logprobs: int,
) -> list[dict[str, Any]]:
    """Convert teacher answers into token-level top-k supervision for distillation."""
    if not answer_records:
        raise ValueError("Cannot build a training dataset without teacher answer records.")

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MATRIX.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MATRIX.model_id,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()

    rows: list[dict[str, Any]] = []
    system_prompt = (
        "Please answer the user's question using only the provided context.\n\n"
        f"<context>\n{corpus_text}\n</context>\n\n"
        "Follow the requested answer format exactly. Do not emit <think> tags or chain-of-thought."
    )
    with torch.inference_mode():
        for answer_record in answer_records:
            user_message = answer_record["user_message"]
            assistant_text = _clean_assistant_text(answer_record["assistant_text"])
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            assistant_token_ids = _assistant_target_token_ids(tokenizer, assistant_text)
            if len(assistant_token_ids) <= 1:
                continue

            input_ids = torch.tensor(
                [prompt_ids + assistant_token_ids[:-1]],
                device=device,
                dtype=torch.long,
            )
            logits = model(input_ids=input_ids).logits
            response_start = len(prompt_ids)
            response_end = response_start + len(assistant_token_ids) - 1
            response_logits = logits[:, response_start - 1 : response_end, :]
            log_probs = torch.log_softmax(response_logits, dim=-1)
            target_ids = torch.tensor(
                [assistant_token_ids],
                device=device,
                dtype=torch.long,
            )
            target_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(0).squeeze(-1)
            top_values, top_indices = torch.topk(log_probs.squeeze(0), k=top_logprobs, dim=-1)

            supervision = []
            # Store only the sparse top-k distribution per position; that is enough for
            # the cartridge trainer while keeping the dataset compact and inspectable.
            for row_idx, token_id in enumerate(assistant_token_ids):
                supervision.append(
                        {
                            "token": tokenizer.decode([token_id], skip_special_tokens=False),
                            "token_id": token_id,
                            "logprob": float(target_log_probs[row_idx]),
                            "source": "hf_teacher",
                            "top_logprobs": [
                                {
                                    "token": tokenizer.decode(
                                        [candidate_id],
                                        skip_special_tokens=False,
                                    ),
                                    "token_id": int(candidate_id),
                                    "logprob": float(candidate_logprob),
                                }
                            for candidate_id, candidate_logprob in zip(
                                top_indices[row_idx].tolist(),
                                top_values[row_idx].tolist(),
                                strict=True,
                            )
                        ],
                    }
                )

            row_stub = {
                "slice_ids": [slice_id],
                "system_prompt": system_prompt,
                "messages": [{"role": "user", "content": user_message}],
                "assistant_token_ids": assistant_token_ids,
            }
            row_hash = stable_hash(row_stub)
            rows.append(
                {
                    "record_id": f"{slice_id}-bootstrap-{row_hash[:12]}",
                    "slice_ids": [slice_id],
                    "system_prompt": system_prompt,
                    "messages": [
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": assistant_text},
                    ],
                    "assistant_token_ids": assistant_token_ids,
                    "assistant_supervision": supervision,
                    "row_hash": row_hash,
                    "metadata": {
                        "question": answer_record["question"],
                        "expected_answer": answer_record.get("expected_answer"),
                        "logprob_source": "hf_teacher",
                    },
                }
            )
    write_jsonl(Path(output_path), rows)
    return rows


def _safe_mean(values: list[float]) -> float | None:
    """Return the arithmetic mean or ``None`` when no values are present."""
    if not values:
        return None
    return float(mean(values))


class SemanticEquivalenceJudge:
    """Judge semantic equivalence with a cheap heuristic plus a strict model fallback."""

    def __init__(self, *, device: str, model_id: str = DEFAULT_MATRIX.model_id) -> None:
        """Load the local model used to evaluate relaxed answer equivalence."""
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            attn_implementation="sdpa",
        )
        self.model.to(device)
        self.model.eval()

    def close(self) -> None:
        """Release model memory held by the judge."""
        del self.model
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    @staticmethod
    def _normalize_answer_text(text: str) -> str:
        """Normalize punctuation/articles away for the cheap equivalence heuristic."""
        cleaned = _clean_assistant_text(text).lower()
        cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
        cleaned = re.sub(r"\b(a|an|the)\b", " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    @classmethod
    def _tokenize_normalized_answer(cls, text: str) -> list[str]:
        """Tokenize normalized text into whitespace-separated pieces."""
        normalized = cls._normalize_answer_text(text)
        return normalized.split() if normalized else []

    @classmethod
    def _contains_token_subsequence(cls, larger: list[str], smaller: list[str]) -> bool:
        """Check whether one normalized token list appears inside another."""
        if not smaller or len(smaller) > len(larger):
            return False
        width = len(smaller)
        return any(larger[i : i + width] == smaller for i in range(len(larger) - width + 1))

    @classmethod
    def _heuristic_equivalent(cls, *, references: list[str], candidate: str) -> bool:
        """Accept obvious wrapper-only variants without spending a model forward pass."""
        candidate_normalized = cls._normalize_answer_text(candidate)
        if not candidate_normalized:
            return False
        candidate_tokens = cls._tokenize_normalized_answer(candidate)
        for reference in references:
            reference_normalized = cls._normalize_answer_text(reference)
            if not reference_normalized:
                continue
            if candidate_normalized == reference_normalized:
                return True
            reference_tokens = cls._tokenize_normalized_answer(reference)
            if (
                cls._contains_token_subsequence(candidate_tokens, reference_tokens)
                and len(candidate_tokens) - len(reference_tokens) <= 10
            ):
                return True
            if (
                cls._contains_token_subsequence(reference_tokens, candidate_tokens)
                and len(reference_tokens) - len(candidate_tokens) <= 3
            ):
                return True
        return False

    def _render_prompt(
        self,
        *,
        question: str,
        references: list[str],
        candidate: str,
    ) -> str:
        """Render the structured SAME/DIFFERENT judging prompt."""
        reference_lines = "\n".join(f"- {item}" for item in references)
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"/no_think\nQuestion:\n{question}\n\n"
                    f"Reference answers:\n{reference_lines}\n\n"
                    f"Candidate answer:\n{candidate}\n\n"
                    "Are the candidate answer and any reference answer semantically the same? "
                    "Reply with exactly SAME or DIFFERENT."
                ),
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )

    def _model_is_equivalent(
        self,
        *,
        question: str,
        references: list[str],
        candidate: str,
    ) -> bool:
        """Fall back to deterministic generation when heuristics are inconclusive."""
        prompt_text = self._render_prompt(
            question=question,
            references=references,
            candidate=candidate,
        )
        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        with torch.inference_mode():
            output_ids = self.model.generate(
                **encoded,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        new_tokens = output_ids[0][encoded["input_ids"].shape[1] :]
        raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        cleaned = _clean_assistant_text(raw_text)
        cleaned = cleaned.replace("<|im_end|>", " ").strip().upper()
        if "DIFFERENT" in cleaned:
            return False
        if "SAME" in cleaned:
            return True
        return False

    def is_equivalent(
        self,
        *,
        question: str,
        references: list[str],
        candidate: str,
    ) -> bool:
        """Return whether a candidate answer is semantically equivalent to any reference."""
        if self._heuristic_equivalent(references=references, candidate=candidate):
            return True
        return self._model_is_equivalent(
            question=question,
            references=references,
            candidate=candidate,
        )


def write_budget_report(
    *,
    experiment_name: str,
    budget_label: str,
    baseline_path: str | Path,
    cartridge_path: str | Path,
    output_dir: str | Path,
    build_seconds: float,
    bootstrap_question_count: int,
    train_steps: int,
    cartridge_tokens: int,
    semantic_judge: bool = False,
    judge_device: str | None = None,
) -> dict[str, Any]:
    """Merge baseline and cartridge predictions into one per-budget report."""
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

    judge = (
        SemanticEquivalenceJudge(device=judge_device or "cpu")
        if semantic_judge
        else None
    )
    paired_rows: list[dict[str, Any]] = []
    for baseline, cartridge in zip(baseline_rows, cartridge_rows, strict=True):
        row = {
            "prompt_id": baseline["prompt_id"],
            "question": baseline["metadata"]["question_id"],
            "query": baseline["metadata"]["query"],
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
            "compression_ratio": baseline["canonical_kv_bytes"] / cartridge["canonical_kv_bytes"],
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
        if judge is not None:
            references = [str(item) for item in row["gold"]]
            row["baseline_semantic_match"] = (
                True
                if row["baseline_exact_match"]
                else judge.is_equivalent(
                    question=row["query"],
                    references=references,
                    candidate=row["baseline_prediction"],
                )
            )
            row["cartridge_semantic_match"] = (
                True
                if row["cartridge_exact_match"]
                else judge.is_equivalent(
                    question=row["query"],
                    references=references,
                    candidate=row["cartridge_prediction"],
                )
            )
        paired_rows.append(row)
    if judge is not None:
        judge.close()

    # The first query is treated separately because cartridge build cost is paid before it.
    baseline_first = paired_rows[0]
    followups = paired_rows[1:]
    summary = {
        "experiment_name": experiment_name,
        "budget_label": budget_label,
        "cartridge_tokens": cartridge_tokens,
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
            [row["throughput_ratio"] for row in paired_rows if row["throughput_ratio"] is not None]
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
    if semantic_judge:
        summary["baseline_semantic_match_rate"] = (
            sum(int(bool(row["baseline_semantic_match"])) for row in paired_rows) / len(paired_rows)
        )
        summary["cartridge_semantic_match_rate"] = (
            sum(int(bool(row["cartridge_semantic_match"])) for row in paired_rows) / len(paired_rows)
        )
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
        f"# {experiment_name} Benchmark ({budget_label})",
        "",
        f"- Questions: {summary['num_questions']}",
        f"- Bootstrap questions: {bootstrap_question_count}",
        f"- Train steps: {train_steps}",
        f"- Cartridge tokens: {cartridge_tokens}",
        f"- One-time compression/build time: {build_seconds:.2f}s",
        f"- Baseline exact-match rate: {summary['baseline_exact_match_rate']:.3f}",
        f"- Cartridge exact-match rate: {summary['cartridge_exact_match_rate']:.3f}",
        (
            f"- Baseline semantic-match rate: {summary['baseline_semantic_match_rate']:.3f}"
            if semantic_judge
            else "- Baseline semantic-match rate: n/a"
        ),
        (
            f"- Cartridge semantic-match rate: {summary['cartridge_semantic_match_rate']:.3f}"
            if semantic_judge
            else "- Cartridge semantic-match rate: n/a"
        ),
        f"- Average compression ratio: {summary['avg_compression_ratio']:.3f}x",
        (
            f"- Average decode throughput ratio: {summary['avg_throughput_ratio']:.3f}x"
            if summary["avg_throughput_ratio"] is not None
            else "- Average decode throughput ratio: n/a"
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
            "| question | baseline_em | cartridge_em | baseline_sem | cartridge_sem | "
            "compression_ratio | decode_ratio | prefill_speedup | e2e_speedup |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in paired_rows:
        throughput = row["throughput_ratio"]
        prefill = row["prefill_speedup_ratio"]
        e2e = row["end_to_end_speedup_ratio"]
        baseline_sem = row.get("baseline_semantic_match")
        cartridge_sem = row.get("cartridge_semantic_match")
        lines.append(
            f"| {row['question']} | {int(row['baseline_exact_match'])} | "
            f"{int(row['cartridge_exact_match'])} | "
            f"{'n/a' if baseline_sem is None else int(bool(baseline_sem))} | "
            f"{'n/a' if cartridge_sem is None else int(bool(cartridge_sem))} | "
            f"{row['compression_ratio']:.3f} | "
            f"{throughput:.3f} | {prefill:.3f} | {e2e:.3f} |"
            if throughput is not None and prefill is not None and e2e is not None
            else
            f"| {row['question']} | {int(row['baseline_exact_match'])} | "
            f"{int(row['cartridge_exact_match'])} | "
            f"{'n/a' if baseline_sem is None else int(bool(baseline_sem))} | "
            f"{'n/a' if cartridge_sem is None else int(bool(cartridge_sem))} | "
            f"{row['compression_ratio']:.3f} | n/a | n/a | n/a |"
        )
    report_path = output_dir / "comparison.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        **summary,
        "comparison_path": str((output_dir / "comparison.jsonl").resolve()),
        "report_path": str(report_path.resolve()),
    }


def write_run_report(
    *,
    experiment_name: str,
    run_dir: str | Path,
    budget_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write the aggregate report that compares multiple cartridge budgets in one run."""
    if not budget_summaries:
        raise ValueError("Cannot write an aggregate run report without budget summaries.")

    report_dir = Path(run_dir) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    semantic_judge_enabled = any(
        "baseline_semantic_match_rate" in item or "cartridge_semantic_match_rate" in item
        for item in budget_summaries
    )
    summary = {
        "experiment_name": experiment_name,
        "semantic_judge_enabled": semantic_judge_enabled,
        "budgets": budget_summaries,
    }
    write_json(report_dir / "summary.json", summary)

    lines = [
        f"# {experiment_name} Benchmark",
        "",
        (
            "| budget | baseline_em | cartridge_em | baseline_sem | cartridge_sem | "
            "compression | decode_ratio | prefill_speedup | e2e_speedup | build_time_s | "
            "baseline_followup_ms | cartridge_followup_ms |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in budget_summaries:
        baseline_semantic = item.get("baseline_semantic_match_rate")
        cartridge_semantic = item.get("cartridge_semantic_match_rate")
        lines.append(
            f"| {item['budget_label']} | {item['baseline_exact_match_rate']:.3f} | "
            f"{item['cartridge_exact_match_rate']:.3f} | "
            f"{'n/a' if baseline_semantic is None else f'{baseline_semantic:.3f}'} | "
            f"{'n/a' if cartridge_semantic is None else f'{cartridge_semantic:.3f}'} | "
            f"{item['avg_compression_ratio']:.3f}x | {item['avg_throughput_ratio']:.3f}x | "
            f"{item['avg_prefill_speedup_ratio']:.3f}x | "
            f"{item['avg_end_to_end_speedup_ratio']:.3f}x | "
            f"{item['compression_build_seconds']:.2f} | "
            f"{item['baseline_followup_total_latency_ms']:.2f} | "
            f"{item['cartridge_followup_total_latency_ms']:.2f} |"
        )
    report_path = report_dir / "comparison.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        **summary,
        "summary_path": str((report_dir / "summary.json").resolve()),
        "report_path": str(report_path.resolve()),
    }
