from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer

from cartridges.config import DEFAULT_MATRIX
from cartridges.core import TrainableKVCartridge, initialize_from_prefix_text
from cartridges.data.common import stable_hash, write_json


@dataclass(frozen=True)
class TrainingExample:
    record_id: str
    slice_id: str
    system_prompt: str
    user_message: str
    assistant_token_ids: list[int]
    assistant_supervision: list[dict[str, Any]]


def load_training_examples(path: str | Path) -> list[TrainingExample]:
    rows: list[TrainingExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                TrainingExample(
                    record_id=row["record_id"],
                    slice_id=row["slice_ids"][0],
                    system_prompt=row["system_prompt"],
                    user_message=row["messages"][0]["content"],
                    assistant_token_ids=[int(token_id) for token_id in row["assistant_token_ids"]],
                    assistant_supervision=row["assistant_supervision"],
                )
            )
    if not rows:
        raise ValueError(f"No training examples found in {path}.")
    return rows


def _training_prompt(tokenizer, user_message: str) -> torch.Tensor:
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    return tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"]


def _sparse_distillation_loss(
    logits: torch.Tensor,
    supervision: list[dict[str, Any]],
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_losses: list[torch.Tensor] = []
    for row_idx, token_supervision in enumerate(supervision):
        candidate_ids: list[int] = []
        candidate_weights: list[float] = []
        for candidate in token_supervision["top_logprobs"]:
            token_id = candidate.get("token_id")
            if token_id is None:
                continue
            candidate_ids.append(int(token_id))
            candidate_weights.append(math.exp(float(candidate["logprob"])))
        target_token_id = int(token_supervision["token_id"])
        if target_token_id not in candidate_ids:
            candidate_ids.append(target_token_id)
            candidate_weights.append(math.exp(float(token_supervision["logprob"])))
        weights = torch.tensor(candidate_weights, device=logits.device, dtype=torch.float32)
        weights = weights / weights.sum()
        token_losses.append(
            -(weights * log_probs[row_idx, torch.tensor(candidate_ids, device=logits.device)]).sum()
        )
    return torch.stack(token_losses).mean()


def _save_checkpoint(
    *,
    path: str | Path,
    cartridge: TrainableKVCartridge,
    optimizer: AdamW,
    scheduler: LambdaLR,
    global_step: int,
    loss_history: list[float],
    metadata: dict[str, Any],
) -> None:
    checkpoint = {
        "num_frozen_tokens": cartridge.num_frozen_tokens,
        "keys": [layer[0].detach().cpu() for layer in cartridge.as_legacy_past_key_values()],
        "values": [layer[1].detach().cpu() for layer in cartridge.as_legacy_past_key_values()],
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
        "loss_history": loss_history,
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "metadata": metadata,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def _load_checkpoint(
    path: str | Path,
    *,
    device: str,
) -> tuple[TrainableKVCartridge, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cartridge = TrainableKVCartridge(
        keys=checkpoint["keys"],
        values=checkpoint["values"],
        num_frozen_tokens=checkpoint["num_frozen_tokens"],
    )
    cartridge.to(device)
    return cartridge, checkpoint


def train_cartridge(
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    slice_id: str | None = None,
    device: str = "cuda:0",
    cartridge_tokens: int = 256,
    num_frozen_tokens: int = 1,
    learning_rate: float = 1e-2,
    steps: int = 20,
    gradient_accumulation_steps: int = 1,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    examples = load_training_examples(dataset_path)
    if slice_id is None:
        slice_id = examples[0].slice_id
    examples = [example for example in examples if example.slice_id == slice_id]
    if not examples:
        raise ValueError(f"No examples found for slice_id={slice_id}.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MATRIX.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MATRIX.model_id,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    checkpoint_metadata = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "slice_id": slice_id,
        "cartridge_tokens": cartridge_tokens,
        "num_frozen_tokens": num_frozen_tokens,
        "model_id": DEFAULT_MATRIX.model_id,
    }

    if resume_from is not None:
        cartridge, checkpoint = _load_checkpoint(resume_from, device=device)
        optimizer = AdamW(cartridge.parameters(), lr=learning_rate)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        random.setstate(checkpoint["python_random_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.startswith("cuda") and checkpoint["cuda_rng_state_all"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        start_step = int(checkpoint["global_step"])
        loss_history = [float(value) for value in checkpoint["loss_history"]]
    else:
        cartridge = initialize_from_prefix_text(
            model=model,
            tokenizer=tokenizer,
            text=examples[0].system_prompt,
            num_tokens=cartridge_tokens,
            num_frozen_tokens=num_frozen_tokens,
        )
        cartridge.to(device)
        optimizer = AdamW(cartridge.parameters(), lr=learning_rate)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        start_step = 0
        loss_history: list[float] = []

    optimizer.zero_grad(set_to_none=True)
    for step_idx in range(start_step, steps):
        example = examples[step_idx % len(examples)]
        prompt_ids = _training_prompt(tokenizer, example.user_message).to(device)
        if len(example.assistant_token_ids) > 1:
            assistant_prefix = torch.tensor(
                [example.assistant_token_ids[:-1]],
                device=device,
                dtype=prompt_ids.dtype,
            )
            model_input = torch.cat([prompt_ids, assistant_prefix], dim=-1)
        else:
            model_input = prompt_ids

        outputs = model(
            input_ids=model_input,
            past_key_values=cartridge.as_cache(model.config),
            use_cache=False,
        )
        target_len = len(example.assistant_token_ids)
        start_idx = prompt_ids.shape[-1] - 1
        end_idx = start_idx + target_len
        assistant_logits = outputs.logits[0, start_idx:end_idx, :]
        loss = _sparse_distillation_loss(assistant_logits, example.assistant_supervision)
        loss_history.append(float(loss.item()))
        (loss / gradient_accumulation_steps).backward()

        should_step = ((step_idx - start_step + 1) % gradient_accumulation_steps) == 0
        if should_step:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    cartridge_path = output_dir / f"{slice_id}_cartridge.pt"
    checkpoint_path = output_dir / f"{slice_id}_checkpoint.pt"
    cartridge.save(cartridge_path)
    _save_checkpoint(
        path=checkpoint_path,
        cartridge=cartridge,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=steps,
        loss_history=loss_history,
        metadata=checkpoint_metadata,
    )

    summary = {
        "slice_id": slice_id,
        "dataset_path": str(Path(dataset_path).resolve()),
        "cartridge_path": str(cartridge_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "steps": steps,
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "loss_history": loss_history,
        "loss_decreased": loss_history[-1] < loss_history[0],
        "manifest_hash": stable_hash(
            {
                "slice_id": slice_id,
                "dataset_path": str(Path(dataset_path).resolve()),
                "cartridge_path": str(cartridge_path.resolve()),
                "checkpoint_path": str(checkpoint_path.resolve()),
                "steps": steps,
                "initial_loss": loss_history[0],
                "final_loss": loss_history[-1],
            }
        ),
    }
    write_json(output_dir / f"{slice_id}_summary.json", summary)
    return summary
