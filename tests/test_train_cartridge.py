from __future__ import annotations

import torch

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
