import json
from unittest.mock import patch

from cartridges.data.text_dataset import build_text_manifest, load_corpus_slices


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        assert text == "corpus"
        return list(range(21))

    def decode(self, token_ids: list[int]) -> str:
        return f"{token_ids[0]}:{token_ids[-1] + 1}"


def test_build_text_manifest_uses_overlap_stride_and_loads_all_slices(tmp_path) -> None:
    source_path = tmp_path / "data.txt"
    manifest_path = tmp_path / "manifest.json"
    source_path.write_text("corpus", encoding="utf-8")

    with patch("cartridges.data.text_dataset.AutoTokenizer") as tokenizer_cls:
        tokenizer_cls.from_pretrained.return_value = _FakeTokenizer()
        manifest = build_text_manifest(
            source_path=source_path,
            output_path=manifest_path,
            chunk_tokens=8,
            stride_tokens=6,
            corpus_id="demo",
        )

    assert manifest["num_chunks"] == 4
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [chunk["start_token"] for chunk in stored["chunks"]] == [0, 6, 12, 13]
    assert [chunk["end_token"] for chunk in stored["chunks"]] == [8, 14, 20, 21]

    slices = load_corpus_slices(manifest_path)
    assert manifest["num_chunks"] == 4
    assert [item["chunk_id"] for item in slices] == ["text-0", "text-6", "text-12", "text-13"]
    assert [item["text"] for item in slices] == ["0:8", "6:14", "12:20", "13:21"]
