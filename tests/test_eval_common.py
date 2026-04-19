from pathlib import Path

from cartridges.eval.cartridge import _clean_completion, _route_for_rows
from cartridges.eval.common import build_cartridge_messages, build_messages


def _row() -> dict[str, object]:
    return {
        "context": "India is a country in South Asia.",
        "query": "Where is India located?",
        "answer_prompt": "Answer with only the region.",
    }


def test_build_messages_includes_context_system_prompt() -> None:
    messages = build_messages(_row())
    assert messages[0]["role"] == "system"
    assert "South Asia" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "Answer with only the region." in messages[1]["content"]


def test_build_cartridge_messages_uses_user_only_prompt() -> None:
    messages = build_cartridge_messages(_row())
    assert messages == [
        {
            "role": "user",
            "content": "/no_think\nWhere is India located?\n\nAnswer with only the region.",
        }
    ]


def test_clean_completion_strips_assistant_prefixes() -> None:
    assert _clean_completion("Assistant: Assistant: The Arabian Sea.") == "The Arabian Sea."


def test_route_for_rows_supports_single_cartridge_without_retrieval_routes() -> None:
    rows = [{"sample_id": "demo", "row_hash": "row-1"}]
    cartridge_paths, routes = _route_for_rows(
        rows=rows,
        cartridge_path=Path("/tmp/cartridge.pt"),
        cartridge_paths=None,
        retrieval_routes=None,
    )

    assert cartridge_paths == {"default": Path("/tmp/cartridge.pt")}
    assert routes["demo::row-1"]["retrieved_slice_id"] == "default"
