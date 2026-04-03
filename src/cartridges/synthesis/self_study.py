from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any

from cartridges.clients import TokenLogprob, TopLogprobCandidate, VLLMClient
from cartridges.config import DEFAULT_MATRIX, RunMode
from cartridges.data.common import stable_hash, write_json, write_jsonl

SEED_PROMPT_FAMILIES = (
    "structuring",
    "summarization",
    "question",
    "use_case",
    "creative",
)

SYSTEM_PROMPT_TEMPLATE = """You are in a conversation about the following corpus excerpt.

<info>
{subcorpus}
</info>

Use the corpus excerpt to answer questions faithfully. Do not emit <think> tags and do not expose chain-of-thought."""

SEED_PROMPT_REGISTRY = {
    "structuring": (
        "Please generate a single chat message instructing an LLM to structure information from "
        "the corpus excerpt into a concrete format such as JSON, YAML, TOML, XML, or plain text. "
        "The message must reference specific sections, entities, or values from the corpus. "
        "Output only the chat message."
    ),
    "summarization": (
        "Please generate a single chat message instructing an LLM to summarize part of the corpus "
        "excerpt. The request should be explicit about what subsection, figure, section, or topic "
        "needs to be summarized. Output only the chat message."
    ),
    "question": (
        "Generate a single user question that tests knowledge of the corpus excerpt. The question "
        "should include enough names, titles, dates, sections, or values to make the target span "
        "clear. Output only the question."
    ),
    "use_case": (
        "Generate a single practical user request that applies the corpus excerpt to a realistic "
        "task, such as analysis, retrieval, planning, or writing. Output only the user message."
    ),
    "creative": (
        "Generate a single creative but corpus-grounded opening message for a conversation about "
        "the excerpt. It can ask for a poem, analogy, dialogue, or other creative transformation, "
        "but it must stay anchored to the corpus content. Output only the user message."
    ),
}


@dataclass(frozen=True)
class CorpusSlice:
    corpus_id: str
    slice_id: str
    text: str
    provenance: dict[str, Any]
    row_hash: str


@dataclass(frozen=True)
class MessageRecord:
    role: str
    content: str


@dataclass(frozen=True)
class AssistantTopLogprobCandidate:
    token: str
    token_id: int | None
    logprob: float


@dataclass(frozen=True)
class AssistantTokenRecord:
    token: str
    token_id: int
    logprob: float
    source: str
    top_logprobs: list[AssistantTopLogprobCandidate]


@dataclass(frozen=True)
class ConversationRecord:
    record_id: str
    corpus_id: str
    slice_ids: list[str]
    provenance: list[dict[str, Any]]
    seed_prompt_family: str
    seed_prompt: str
    system_prompt: str
    messages: list[MessageRecord]
    assistant_token_ids: list[int]
    assistant_supervision: list[AssistantTokenRecord]
    model_id: str
    model_revision: str | None
    tokenizer_id: str
    tokenizer_revision: str | None
    usage: dict[str, int] | None
    finish_reason: str | None
    logprob_source: str
    row_hash: str


def _token_logprob_to_dict(token: TokenLogprob) -> AssistantTokenRecord:
    return AssistantTokenRecord(
        token=token.token,
        token_id=token.token_id,
        logprob=token.logprob,
        source=token.source,
        top_logprobs=[
            AssistantTopLogprobCandidate(
                token=candidate.token,
                token_id=candidate.token_id,
                logprob=candidate.logprob,
            )
            for candidate in token.top_logprobs
        ],
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_corpus_slices(path: str | Path) -> list[CorpusSlice]:
    payload = _load_json(path)
    corpus_id = payload.get("corpus_id", Path(path).stem)

    if "chunks" in payload:
        return [
            CorpusSlice(
                corpus_id=corpus_id,
                slice_id=chunk["chunk_id"],
                text=chunk["text"],
                provenance={
                    "start_token": chunk["start_token"],
                    "end_token": chunk["end_token"],
                },
                row_hash=chunk["row_hash"],
            )
            for chunk in payload["chunks"]
        ]

    if "samples" in payload:
        return [
            CorpusSlice(
                corpus_id=corpus_id,
                slice_id=sample["sample_id"],
                text=sample["context"],
                provenance={
                    "expected": sample.get("expected", {}),
                    "query_count": len(sample.get("queries", [])),
                },
                row_hash=sample["row_hash"],
            )
            for sample in payload["samples"]
        ]

    raise ValueError(f"Unsupported resource schema in {path}.")


def _sample_slices(
    slices: list[CorpusSlice],
    rng: random.Random,
    max_context_slices: int,
) -> list[CorpusSlice]:
    if not slices:
        raise ValueError("Cannot sample from an empty corpus.")
    if max_context_slices <= 1:
        return [rng.choice(slices)]

    count = rng.randint(1, min(max_context_slices, len(slices)))
    return rng.sample(slices, count)


def _build_context_text(selected_slices: list[CorpusSlice]) -> str:
    rendered: list[str] = []
    for item in selected_slices:
        rendered.append(f"[source={item.slice_id}]\n{item.text}")
    return "\n\n".join(rendered)


def _build_record(
    *,
    selected_slices: list[CorpusSlice],
    seed_prompt_family: str,
    seed_prompt: str,
    system_prompt: str,
    user_message: str,
    assistant_message: str,
    assistant_token_ids: list[int],
    assistant_supervision: list[AssistantTokenRecord],
    usage: dict[str, int] | None,
    finish_reason: str | None,
    logprob_source: str,
    client: VLLMClient,
) -> ConversationRecord:
    row_without_hash = {
        "corpus_id": selected_slices[0].corpus_id,
        "slice_ids": [item.slice_id for item in selected_slices],
        "seed_prompt_family": seed_prompt_family,
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ],
        "assistant_token_ids": assistant_token_ids,
        "assistant_supervision": [asdict(item) for item in assistant_supervision],
    }
    row_hash = stable_hash(row_without_hash)
    return ConversationRecord(
        record_id=f"{selected_slices[0].corpus_id}-{row_hash[:12]}",
        corpus_id=selected_slices[0].corpus_id,
        slice_ids=[item.slice_id for item in selected_slices],
        provenance=[
            {
                "slice_id": item.slice_id,
                "row_hash": item.row_hash,
                **item.provenance,
            }
            for item in selected_slices
        ],
        seed_prompt_family=seed_prompt_family,
        seed_prompt=seed_prompt,
        system_prompt=system_prompt,
        messages=[
            MessageRecord(role="user", content=user_message),
            MessageRecord(role="assistant", content=assistant_message),
        ],
        assistant_token_ids=assistant_token_ids,
        assistant_supervision=assistant_supervision,
        model_id=client.model_id,
        model_revision=client.model_revision,
        tokenizer_id=client.tokenizer_id,
        tokenizer_revision=client.tokenizer_revision,
        usage=usage,
        finish_reason=finish_reason,
        logprob_source=logprob_source,
        row_hash=row_hash,
    )


def validate_record(record: ConversationRecord) -> None:
    if record.seed_prompt_family not in SEED_PROMPT_FAMILIES:
        raise ValueError(f"Unsupported seed prompt family: {record.seed_prompt_family}")
    if len(record.messages) != 2:
        raise ValueError("ConversationRecord must contain exactly one user and one assistant message.")
    if record.messages[0].role != "user" or record.messages[1].role != "assistant":
        raise ValueError("ConversationRecord messages must be ordered as user, assistant.")
    if not record.messages[0].content.strip():
        raise ValueError("User message is empty.")
    if not record.messages[1].content.strip():
        raise ValueError("Assistant message is empty.")
    if not record.assistant_token_ids:
        raise ValueError("Assistant token ids are empty.")
    if len(record.assistant_supervision) != len(record.assistant_token_ids):
        raise ValueError("Assistant supervision length does not match token ids length.")
    for token in record.assistant_supervision:
        if not token.top_logprobs:
            raise ValueError("Assistant supervision contains a token without top-logprobs.")


def run_self_study_synthesis(
    *,
    resource_path: str | Path,
    output_dir: str | Path,
    base_url: str,
    api_key: str,
    run_mode: RunMode,
    num_samples: int,
    seed: int = 42,
    top_logprobs: int = 20,
    temperature_a: float = 0.8,
    temperature_b: float = 0.0,
    max_completion_tokens_a: int = 128,
    max_completion_tokens_b: int = 384,
    max_context_slices: int = 1,
    teacher_device: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slices = load_corpus_slices(resource_path)
    rng = random.Random(seed)
    client = VLLMClient(
        base_url=base_url,
        api_key=api_key,
        model_id=DEFAULT_MATRIX.model_id,
        teacher_device=teacher_device,
    )
    records: list[ConversationRecord] = []

    try:
        parity = client.probe_tokenizer_parity()
        if not parity.matches:
            raise RuntimeError(
                f"Tokenizer mismatch between local HF and vLLM: {parity.local_token_ids} vs {parity.server_token_ids}"
            )

        for _ in range(num_samples):
            selected_slices = _sample_slices(slices, rng=rng, max_context_slices=max_context_slices)
            corpus_text = _build_context_text(selected_slices)
            seed_prompt_family = rng.choice(SEED_PROMPT_FAMILIES)
            seed_prompt = SEED_PROMPT_REGISTRY[seed_prompt_family]
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(subcorpus=corpus_text)

            user_turn = client.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": seed_prompt},
                ],
                max_completion_tokens=max_completion_tokens_a,
                temperature=temperature_a,
                run_mode=run_mode,
            )
            user_message = user_turn.text.strip()
            if not user_message:
                raise RuntimeError("Self-study user generation returned an empty message.")

            assistant_turn = client.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_completion_tokens=max_completion_tokens_b,
                temperature=temperature_b,
                top_logprobs=top_logprobs,
                run_mode=run_mode,
            )
            assistant_supervision = [
                _token_logprob_to_dict(token) for token in assistant_turn.token_logprobs
            ]
            record = _build_record(
                selected_slices=selected_slices,
                seed_prompt_family=seed_prompt_family,
                seed_prompt=seed_prompt,
                system_prompt=system_prompt,
                user_message=user_message,
                assistant_message=assistant_turn.text.strip(),
                assistant_token_ids=assistant_turn.token_ids,
                assistant_supervision=assistant_supervision,
                usage=assistant_turn.usage,
                finish_reason=assistant_turn.finish_reason,
                logprob_source=assistant_turn.logprob_source,
                client=client,
            )
            validate_record(record)
            records.append(record)
    finally:
        client.close()

    rows = [asdict(record) for record in records]
    conversations_path = output_dir / "conversations.jsonl"
    write_jsonl(conversations_path, rows)

    manifest = {
        "resource_path": str(Path(resource_path).resolve()),
        "output_dir": str(output_dir.resolve()),
        "run_mode": run_mode,
        "num_samples": num_samples,
        "seed": seed,
        "top_logprobs": top_logprobs,
        "max_context_slices": max_context_slices,
        "model_id": DEFAULT_MATRIX.model_id,
        "records_path": conversations_path.name,
        "record_hashes": [record.row_hash for record in records],
        "tokenizer_parity": True,
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    write_json(output_dir / "manifest.json", manifest)
    return manifest
