"""Distill teacher responses into a trainable KV cartridge."""

import json
import math
import random
from copy import deepcopy
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
    """One distillation row: chat prompt + target completion + sparse teacher distributions.

    Loaded from JSONL written by ``build_training_dataset`` (``text_benchmark``). Fields
    mirror that schema:

    - ``system_prompt`` / ``user_message``: first message roles in the stored row;
      training seeds the cartridge from ``system_prompt`` and builds the student prompt
      from ``user_message`` only (see ``_training_prompt``), matching routed inference.
    - ``assistant_token_ids``: target completion as tokenizer ids for the assistant
      answer text plus a final EOS (see ``_assistant_target_token_ids`` in
      ``text_benchmark``). Length equals ``len(assistant_supervision)``.
    - ``assistant_supervision``: one dict per target token with ``token_id``, teacher
      ``logprob``, and ``top_logprobs`` (sparse top-k over the HF teacher); produced in
      ``build_training_dataset`` and consumed by ``_sparse_distillation_loss``.
    """
    record_id: str
    slice_id: str
    system_prompt: str
    user_message: str
    assistant_token_ids: list[int]
    assistant_supervision: list[dict[str, Any]]


def _set_training_seed(seed: int) -> None:
    """Set Python and Torch RNGs so repeated runs stay comparable."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_training_examples(path: str | Path) -> list[TrainingExample]:
    """Load JSONL lines from ``build_training_dataset`` into ``TrainingExample`` records."""
    rows: list[TrainingExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            # JSONL keys align with rows appended in ``build_training_dataset``:
            # ``messages[0]`` is the user turn; assistant text is redundant with token ids.
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
    """Tokenize the user-side prompt exactly as it will appear at inference time."""
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
    """Cross-entropy against the sparse teacher distribution stored in ``assistant_supervision``.

    ``supervision`` comes from ``build_training_dataset``. For each assistant token position it
    stores:

    - ``token_id`` / ``logprob``: the teacher probability assigned to the actual target token
      that appeared in the reference answer.
    - ``top_logprobs``: a top-k sparse snapshot of the teacher distribution at that same step.
      Each entry is a likely alternative token with its own ``token_id`` and ``logprob``.

    The loss treats those sparse probabilities as a normalized teacher distribution and computes
    cross-entropy against the student's logits for that token position.
    """
    # Convert student logits into log-probabilities once up front. We keep the result in log
    # space because the final loss is a weighted negative log-likelihood.
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_losses: list[torch.Tensor] = []
    for row_idx, token_supervision in enumerate(supervision):
        # ``candidate_ids`` are the vocabulary ids that define the sparse teacher distribution for
        # this output step. ``candidate_weights`` are the corresponding probabilities in normal
        # probability space, reconstructed from stored logprobs.
        candidate_ids: list[int] = []
        candidate_weights: list[float] = []
        # ``top_logprobs`` is the teacher's top-k token list for this single decoding step.
        # It is sparse: only the most likely alternatives are stored, not the full vocabulary.
        for candidate in token_supervision["top_logprobs"]:
            token_id = candidate.get("token_id")
            if token_id is None:
                continue
            candidate_ids.append(int(token_id))
            # The JSONL stores log-probabilities because they are stable and compact. Convert back
            # to probabilities here so we can normalize the sparse teacher mass.
            candidate_weights.append(math.exp(float(candidate["logprob"])))
        # ``token_id`` is the actual reference token for this step. In principle it should already
        # appear in ``top_logprobs``. If the teacher assigned it low enough probability that it
        # fell out of the top-k, we add it back explicitly so the gold token is never unsupervised.
        target_token_id = int(token_supervision["token_id"])
        if target_token_id not in candidate_ids:
            candidate_ids.append(target_token_id)
            candidate_weights.append(math.exp(float(token_supervision["logprob"])))
        # Re-normalize because ``top_logprobs`` covers only a truncated slice of the full teacher
        # distribution. This turns the kept candidates into a proper categorical distribution.
        weights = torch.tensor(candidate_weights, device=logits.device, dtype=torch.float32)
        weights = weights / weights.sum()
        # Gather the student's log-probabilities only at the sparse candidate ids instead of over
        # the full vocabulary. The weighted sum below is:
        #
        #   -Σ_i teacher_prob_i * log student_prob_i
        #
        # which is the cross-entropy from the sparse teacher distribution to the student.
        token_losses.append(
            -(weights * log_probs[row_idx, torch.tensor(candidate_ids, device=logits.device)]).sum()
        )
    # Average over assistant token positions so every training example contributes one scalar loss.
    return torch.stack(token_losses).mean()


def _compute_example_loss(
    *,
    model,
    tokenizer,
    cartridge: TrainableKVCartridge,
    example: TrainingExample,
    device: str,
) -> torch.Tensor:
    """Compute the sparse distillation loss for one prompt/answer pair.

    The example already contains the teacher-aligned assistant target tokens and per-token sparse
    supervision. This function's job is just to reproduce the same token positions on the student
    side so ``_sparse_distillation_loss`` compares like with like.
    """
    prompt_ids = _training_prompt(tokenizer, example.user_message).to(device)
    # Teacher-forcing: append all target tokens except the last so each logits position
    # predicts the next ``assistant_token_ids`` entry (same alignment as in
    # ``build_training_dataset`` when slicing ``response_logits``).
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
    # The logit at position ``prompt_len - 1`` predicts the first assistant token; after that the
    # teacher-forced assistant prefix makes each subsequent row predict the next target token.
    # The slice below therefore aligns 1:1 with ``assistant_supervision`` / ``assistant_token_ids``.
    start_idx = prompt_ids.shape[-1] - 1
    end_idx = start_idx + target_len
    assistant_logits = outputs.logits[0, start_idx:end_idx, :]
    return _sparse_distillation_loss(assistant_logits, example.assistant_supervision)


def _evaluate_examples_loss(
    *,
    model,
    tokenizer,
    cartridge: TrainableKVCartridge,
    examples: list[TrainingExample],
    device: str,
) -> float:
    """Average loss over a fixed validation subset for stable checkpoint selection."""
    if not examples:
        raise ValueError("Validation examples must not be empty.")
    losses: list[float] = []
    with torch.inference_mode():
        for example in examples:
            loss = _compute_example_loss(
                model=model,
                tokenizer=tokenizer,
                cartridge=cartridge,
                example=example,
                device=device,
            )
            losses.append(float(loss.item()))
    return sum(losses) / len(losses)


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
    """Persist cartridge weights together with optimizer and RNG state for resume."""
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
    """Reload a saved training checkpoint into a live cartridge object."""
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
    learning_rate: float = 3e-3,
    steps: int = 20,
    gradient_accumulation_steps: int = 1,
    resume_from: str | Path | None = None,
    max_grad_norm: float = 1.0,
    seed: int = 0,
    validation_examples: int = 16,
    validation_interval: int = 10,
) -> dict[str, Any]:
    """Train one cartridge budget against a fixed supervision dataset.

    The cartridge starts from a truncated prefix KV cache, but the supervision dataset was
    produced from the full chunk context. Optimization therefore teaches the small cartridge
    to approximate full-context behavior rather than to stay a literal cache for only the
    first ``cartridge_tokens`` input tokens.
    """
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if validation_examples <= 0:
        raise ValueError("validation_examples must be positive.")
    if validation_interval <= 0:
        raise ValueError("validation_interval must be positive.")

    examples = load_training_examples(dataset_path)
    if slice_id is None:
        slice_id = examples[0].slice_id
    examples = [example for example in examples if example.slice_id == slice_id]
    if not examples:
        raise ValueError(f"No examples found for slice_id={slice_id}.")
    validation_subset = examples[: min(validation_examples, len(examples))]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if resume_from is None:
        _set_training_seed(seed)

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
        "seed": seed,
        "validation_examples": len(validation_subset),
        "validation_interval": validation_interval,
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
        # Seed the cartridge from the first ``cartridge_tokens`` positions of the chunk-level
        # system prompt. This is only an initialization: the trainable K/V tensors below are
        # subsequently optimized against supervision that came from the full chunk context.
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
    best_loss = float("inf")
    best_train_loss = float("inf")
    best_step = start_step
    best_state_dict = deepcopy(cartridge.state_dict())
    validation_history: list[dict[str, float | int]] = []
    for step_idx in range(start_step, steps):
        example = examples[step_idx % len(examples)]
        loss = _compute_example_loss(
            model=model,
            tokenizer=tokenizer,
            cartridge=cartridge,
            example=example,
            device=device,
        )
        loss_value = float(loss.item())
        loss_history.append(loss_value)
        # Gradients flow only into ``cartridge.parameters()``. The parent model weights were
        # frozen above, so backprop updates the trainable K/V slots rather than the LLM itself.
        (loss / gradient_accumulation_steps).backward()

        should_step = ((step_idx - start_step + 1) % gradient_accumulation_steps) == 0
        if should_step:
            # Cartridge optimization is high variance; clipping plus periodic validation
            # makes checkpoint selection much more stable than a single minibatch minimum.
            torch.nn.utils.clip_grad_norm_(cartridge.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            completed_step = step_idx + 1
            should_validate = (
                completed_step == steps
                or (completed_step % validation_interval) == 0
            )
            if should_validate:
                validation_loss = _evaluate_examples_loss(
                    model=model,
                    tokenizer=tokenizer,
                    cartridge=cartridge,
                    examples=validation_subset,
                    device=device,
                )
                validation_history.append(
                    {
                        "step": completed_step,
                        "validation_loss": validation_loss,
                        "recent_train_loss": loss_value,
                    }
                )
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    best_train_loss = loss_value
                    best_step = completed_step
                    best_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in cartridge.state_dict().items()
                    }

    final_cartridge_path = output_dir / f"{slice_id}_final_cartridge.pt"
    cartridge_path = output_dir / f"{slice_id}_cartridge.pt"
    checkpoint_path = output_dir / f"{slice_id}_checkpoint.pt"
    cartridge.save(final_cartridge_path)
    cartridge.load_state_dict(best_state_dict)
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
        "final_cartridge_path": str(final_cartridge_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "steps": steps,
        "initial_loss": loss_history[0],
        "best_loss": best_loss,
        "best_train_loss": best_train_loss,
        "best_step": best_step,
        "final_loss": loss_history[-1],
        "loss_history": loss_history,
        "validation_history": validation_history,
        "loss_decreased": loss_history[-1] < loss_history[0],
        "manifest_hash": stable_hash(
            {
                "slice_id": slice_id,
                "dataset_path": str(Path(dataset_path).resolve()),
                "cartridge_path": str(cartridge_path.resolve()),
                "checkpoint_path": str(checkpoint_path.resolve()),
                "steps": steps,
                "seed": seed,
                "initial_loss": loss_history[0],
                "final_loss": loss_history[-1],
            }
        ),
    }
    write_json(output_dir / f"{slice_id}_summary.json", summary)
    return summary
