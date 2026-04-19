import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cartridges.core.cartridge import TrainableKVCartridge

TINY_MODEL_ID = "hf-internal-testing/tiny-random-gpt2"


def _build_cartridge() -> TrainableKVCartridge:
    keys = [torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8)]
    values = [torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8)]
    return TrainableKVCartridge(keys=keys, values=values, num_frozen_tokens=1)


def _load_tiny_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL_ID)
    model.eval()
    return model, tokenizer


def test_canonical_kv_bytes_matches_tensor_storage() -> None:
    cartridge = _build_cartridge()
    expected = 0
    for key, value in cartridge.as_legacy_past_key_values():
        expected += key.numel() * key.element_size()
        expected += value.numel() * value.element_size()
    assert cartridge.canonical_kv_bytes() == expected


def test_save_and_load_round_trip() -> None:
    cartridge = _build_cartridge()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "cartridge.pt"
        cartridge.save(path)
        loaded = TrainableKVCartridge.load(path)

    assert loaded.num_layers == cartridge.num_layers
    assert loaded.num_tokens == cartridge.num_tokens
    for (orig_key, orig_value), (new_key, new_value) in zip(
        cartridge.as_legacy_past_key_values(),
        loaded.as_legacy_past_key_values(),
        strict=True,
    ):
        assert torch.equal(orig_key, new_key)
        assert torch.equal(orig_value, new_value)


def test_only_trainable_suffix_receives_gradients() -> None:
    cartridge = _build_cartridge()
    loss = sum(
        key.square().mean() + value.square().mean()
        for key, value in cartridge.as_legacy_past_key_values()
    )
    loss.backward()

    assert cartridge.trainable_keys[0].grad is not None
    assert cartridge.trainable_values[0].grad is not None
    assert cartridge.frozen_keys[0].grad is None
    assert cartridge.frozen_values[0].grad is None


def test_prefix_cache_matches_literal_prefix_logits() -> None:
    model, tokenizer = _load_tiny_model_and_tokenizer()
    prefix_ids = tokenizer("Alpha beta gamma delta", return_tensors="pt", add_special_tokens=False)["input_ids"]
    query_ids = tokenizer(" epsilon zeta", return_tensors="pt", add_special_tokens=False)["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids=prefix_ids, use_cache=True)
    cartridge = TrainableKVCartridge(
        keys=[layer[0] for layer in outputs.past_key_values.to_legacy_cache()],
        values=[layer[1] for layer in outputs.past_key_values.to_legacy_cache()],
        num_frozen_tokens=1,
    )

    with torch.no_grad():
        full_logits = model(
            input_ids=torch.cat([prefix_ids, query_ids], dim=-1),
            use_cache=False,
        ).logits[:, -query_ids.shape[-1] :, :]
        cache_logits = model(
            input_ids=query_ids,
            past_key_values=cartridge.as_cache(model.config),
            use_cache=False,
        ).logits

    assert torch.allclose(full_logits, cache_logits, atol=1e-5, rtol=1e-4)


def test_model_stays_frozen_with_cache_integration() -> None:
    model, tokenizer = _load_tiny_model_and_tokenizer()
    for param in model.parameters():
        param.requires_grad_(False)

    prefix_ids = tokenizer("Alpha beta gamma delta", return_tensors="pt", add_special_tokens=False)["input_ids"]
    query_ids = tokenizer(" epsilon zeta", return_tensors="pt", add_special_tokens=False)["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids=prefix_ids, use_cache=True)
    cartridge = TrainableKVCartridge(
        keys=[layer[0] for layer in outputs.past_key_values.to_legacy_cache()],
        values=[layer[1] for layer in outputs.past_key_values.to_legacy_cache()],
        num_frozen_tokens=1,
    )

    loss = model(
        input_ids=query_ids,
        past_key_values=cartridge.as_cache(model.config),
        use_cache=False,
    ).logits.float().square().mean()
    loss.backward()

    assert any(param.grad is not None for param in cartridge.trainable_keys)
    assert any(param.grad is not None for param in cartridge.trainable_values)
    assert all(param.grad is None for param in model.parameters())
