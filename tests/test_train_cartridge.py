import json
from pathlib import Path
from unittest.mock import patch
import math

import torch

from cartridges.train import cartridge as train_module
from cartridges.train.cartridge import _training_prompt


class _FakeTokenizer:
    def __init__(self) -> None:
        self.last_kwargs = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, chat_template_kwargs):
        self.last_kwargs = {
            "messages": messages,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "chat_template_kwargs": chat_template_kwargs,
        }
        return "prompt"

    def __call__(self, prompt_text, return_tensors, add_special_tokens):
        assert prompt_text == "prompt"
        assert return_tensors == "pt"
        assert add_special_tokens is False
        return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}


def test_training_prompt_disables_thinking() -> None:
    tokenizer = _FakeTokenizer()
    prompt_ids = _training_prompt(tokenizer, "/no_think\nQuestion")
    assert torch.equal(prompt_ids, torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert tokenizer.last_kwargs == {
        "messages": [{"role": "user", "content": "/no_think\nQuestion"}],
        "tokenize": False,
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


class _TinyTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, chat_template_kwargs):
        del messages, tokenize, add_generation_prompt, chat_template_kwargs
        return "prompt"

    def __call__(self, prompt_text, return_tensors, add_special_tokens):
        del prompt_text, return_tensors, add_special_tokens
        return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type(
            "Config",
            (),
            {
                "num_hidden_layers": 1,
                "num_key_value_heads": 1,
                "hidden_size": 4,
                "num_attention_heads": 1,
                "head_dim": 4,
            },
        )()
        self.dtype = torch.float32
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        return self

    def parameters(self):
        return []

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        del input_ids, past_key_values, use_cache
        logits = torch.zeros((1, 1, 200), dtype=torch.float32)
        logits[..., 7] = 1.0
        return type("Output", (), {"logits": logits})()


class _TinyCartridge(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.num_frozen_tokens = 0

    def as_cache(self, model_config):
        del model_config
        return None

    def as_legacy_past_key_values(self):
        key = self.weight.detach().view(1, 1, 1, 1)
        value = self.weight.detach().view(1, 1, 1, 1)
        return ((key, value),)

    def save(self, path: str | Path) -> None:
        torch.save({"weight": float(self.weight.detach().cpu().item())}, path)


def test_train_cartridge_saves_best_checkpoint(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    row = {
        "record_id": "r1",
        "slice_ids": ["slice-0"],
        "system_prompt": "system",
        "messages": [{"role": "user", "content": "/no_think\nQuestion"}],
        "assistant_token_ids": [7, 99],
        "assistant_supervision": [
            {"token_id": 7, "logprob": 0.0, "top_logprobs": [{"token_id": 7, "logprob": 0.0}]},
            {"token_id": 99, "logprob": 0.0, "top_logprobs": [{"token_id": 99, "logprob": 0.0}]},
        ],
    }
    dataset_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    losses = iter([0.5, 0.1, 0.4])
    validation_losses = iter([0.4, 0.2, 0.3])

    def fake_loss(_logits, _supervision):
        value = next(losses)
        return torch.tensor(value, dtype=torch.float32, requires_grad=True)

    def fake_validation_loss(*, model, tokenizer, cartridge, examples, device):
        del model, tokenizer, cartridge, examples, device
        return next(validation_losses)

    def fake_step(self):
        for group in self.param_groups:
            for param in group["params"]:
                param.data.add_(1.0)

    def fake_zero_grad(self, set_to_none=True):
        del set_to_none

    def fake_scheduler_step(self):
        return None

    def fake_initialize_from_prefix_text(*args, **kwargs):
        del args, kwargs
        return _TinyCartridge()

    with (
        patch.object(train_module, "AutoTokenizer") as tokenizer_cls,
        patch.object(train_module, "AutoModelForCausalLM") as model_cls,
        patch.object(train_module, "initialize_from_prefix_text", side_effect=fake_initialize_from_prefix_text),
        patch.object(train_module, "_sparse_distillation_loss", side_effect=fake_loss),
        patch.object(train_module, "_evaluate_examples_loss", side_effect=fake_validation_loss),
        patch.object(train_module.AdamW, "step", fake_step),
        patch.object(train_module.AdamW, "zero_grad", fake_zero_grad),
        patch.object(train_module.LambdaLR, "step", fake_scheduler_step),
    ):
        tokenizer_cls.from_pretrained.return_value = _TinyTokenizer()
        model_cls.from_pretrained.return_value = _TinyModel()
        summary = train_module.train_cartridge(
            dataset_path=dataset_path,
            output_dir=tmp_path / "out",
            device="cpu",
            steps=3,
            validation_interval=1,
        )

    best_cartridge = torch.load(summary["cartridge_path"], map_location="cpu", weights_only=False)
    final_cartridge = torch.load(summary["final_cartridge_path"], map_location="cpu", weights_only=False)
    assert summary["best_step"] == 2
    assert math.isclose(summary["best_loss"], 0.2, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(summary["best_train_loss"], 0.1, rel_tol=0.0, abs_tol=1e-6)
    assert best_cartridge["weight"] == 2.0
    assert final_cartridge["weight"] == 3.0
