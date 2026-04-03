from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from transformers.cache_utils import DynamicCache


@dataclass(frozen=True)
class AttentionShape:
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    dtype_bytes: int = 2


def _infer_attention_shape(model) -> AttentionShape:
    config = model.config
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    return AttentionShape(
        num_hidden_layers=config.num_hidden_layers,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=head_dim,
    )


def _normalize_past_key_values(past_key_values) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    return [(layer[0], layer[1]) for layer in past_key_values]


class TrainableKVCartridge(nn.Module):
    def __init__(
        self,
        keys: Iterable[torch.Tensor],
        values: Iterable[torch.Tensor],
        num_frozen_tokens: int = 1,
    ):
        super().__init__()
        keys = [tensor.detach().clone() for tensor in keys]
        values = [tensor.detach().clone() for tensor in values]
        if len(keys) != len(values):
            raise ValueError("keys and values must have the same number of layers.")

        self.num_layers = len(keys)
        self.num_tokens = keys[0].shape[-2]
        self.num_frozen_tokens = num_frozen_tokens
        self.trainable_keys = nn.ParameterList()
        self.trainable_values = nn.ParameterList()
        self.frozen_keys = nn.ParameterList()
        self.frozen_values = nn.ParameterList()

        for key_tensor, value_tensor in zip(keys, values, strict=True):
            if num_frozen_tokens > 0:
                frozen_key = nn.Parameter(key_tensor[..., :num_frozen_tokens, :], requires_grad=False)
                frozen_value = nn.Parameter(value_tensor[..., :num_frozen_tokens, :], requires_grad=False)
                train_key = nn.Parameter(key_tensor[..., num_frozen_tokens:, :])
                train_value = nn.Parameter(value_tensor[..., num_frozen_tokens:, :])
                self.frozen_keys.append(frozen_key)
                self.frozen_values.append(frozen_value)
            else:
                train_key = nn.Parameter(key_tensor)
                train_value = nn.Parameter(value_tensor)
            self.trainable_keys.append(train_key)
            self.trainable_values.append(train_value)

    @property
    def num_trainable_tokens(self) -> int:
        return self.num_tokens - self.num_frozen_tokens

    def layer(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        key_parts = []
        value_parts = []
        if self.num_frozen_tokens > 0:
            key_parts.append(self.frozen_keys[index])
            value_parts.append(self.frozen_values[index])
        key_parts.append(self.trainable_keys[index])
        value_parts.append(self.trainable_values[index])
        return torch.cat(key_parts, dim=-2), torch.cat(value_parts, dim=-2)

    def as_legacy_past_key_values(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple(self.layer(index) for index in range(self.num_layers))

    def as_cache(self, model_config) -> DynamicCache:
        return DynamicCache(ddp_cache_data=self.as_legacy_past_key_values(), config=model_config)

    def canonical_kv_bytes(self) -> int:
        total = 0
        for key, value in self.as_legacy_past_key_values():
            total += key.numel() * key.element_size()
            total += value.numel() * value.element_size()
        return total

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "num_frozen_tokens": self.num_frozen_tokens,
                "keys": [tensor.detach().cpu() for tensor in self.trainable_keys],
                "values": [tensor.detach().cpu() for tensor in self.trainable_values],
                "frozen_keys": [tensor.detach().cpu() for tensor in self.frozen_keys],
                "frozen_values": [tensor.detach().cpu() for tensor in self.frozen_values],
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device | None = None) -> "TrainableKVCartridge":
        checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
        keys = []
        values = []
        num_frozen_tokens = checkpoint["num_frozen_tokens"]
        for frozen_key, train_key, frozen_value, train_value in zip(
            checkpoint["frozen_keys"],
            checkpoint["keys"],
            checkpoint["frozen_values"],
            checkpoint["values"],
            strict=True,
        ):
            key = torch.cat([frozen_key, train_key], dim=-2) if num_frozen_tokens else train_key
            value = torch.cat([frozen_value, train_value], dim=-2) if num_frozen_tokens else train_value
            keys.append(key)
            values.append(value)
        cartridge = cls(keys=keys, values=values, num_frozen_tokens=num_frozen_tokens)
        if device is not None:
            cartridge.to(device)
        return cartridge


@torch.no_grad()
def initialize_from_prefix_text(
    model,
    tokenizer,
    text: str,
    num_tokens: int,
    num_frozen_tokens: int = 1,
) -> TrainableKVCartridge:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"][..., :num_tokens].to(model.device)
    outputs = model(input_ids=input_ids, use_cache=True)
    past_key_values = _normalize_past_key_values(outputs.past_key_values)
    keys = [layer[0].detach().to(model.dtype) for layer in past_key_values]
    values = [layer[1].detach().to(model.dtype) for layer in past_key_values]
    return TrainableKVCartridge(keys=keys, values=values, num_frozen_tokens=num_frozen_tokens)
