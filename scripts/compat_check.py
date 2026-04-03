#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.config import DEFAULT_MATRIX, TOKENIZER_PROBE_TEXT  # noqa: E402


def _choose_device(device_arg: str | None) -> str:
    if device_arg is not None:
        return device_arg
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        first = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0].strip()
        return f"cuda:{first}"
    return "cuda:0"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run compatibility checks for the selected stack.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--base-url", default=os.environ.get("CARTRIDGES_VLLM_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "compat-check"))
    parser.add_argument("--skip-vllm-runtime", action="store_true")
    args = parser.parse_args()

    import torch
    from openai import OpenAI
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _choose_device(args.device)
    model_id = DEFAULT_MATRIX.model_id

    print(f"compatibility_matrix={DEFAULT_MATRIX}")
    print(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoded = tokenizer(TOKENIZER_PROBE_TEXT, return_tensors="pt")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.to(device)
    for param in model.parameters():
        param.requires_grad_(False)

    embed = model.get_input_embeddings()(encoded["input_ids"].to(model.device))
    prefix = torch.nn.Parameter(
        0.01 * torch.randn((1, 2, embed.shape[-1]), device=model.device, dtype=model.dtype)
    )
    joined = torch.cat([prefix, embed], dim=1)

    outputs = model(inputs_embeds=joined, use_cache=False)
    loss = outputs.logits.float().square().mean()
    loss.backward()

    if prefix.grad is None or not torch.isfinite(prefix.grad).all() or prefix.grad.abs().sum().item() == 0:
        raise RuntimeError("Gradient did not flow to the compatibility-check cartridge parameter.")
    if any(param.grad is not None for param in model.parameters()):
        raise RuntimeError("Base model parameters received gradients during the compatibility check.")

    print(f"loss={loss.item():.6f}")
    print(f"prefix_grad_norm={prefix.grad.norm().item():.6f}")

    if args.skip_vllm_runtime or not args.base_url:
        print("vllm_runtime=deferred")
        return 0

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": "Reply with the word OK."}],
        max_completion_tokens=4,
        temperature=0.0,
        logprobs=True,
        top_logprobs=2,
    )

    choice = response.choices[0]
    if not choice.logprobs or not choice.logprobs.content:
        raise RuntimeError("vLLM response did not include chat logprobs content.")

    probe_ids = tokenizer.encode(TOKENIZER_PROBE_TEXT, add_special_tokens=False)
    if tokenizer.decode(probe_ids) == "":
        raise RuntimeError("Tokenizer probe unexpectedly decoded to an empty string.")

    print("vllm_runtime=ok")
    print(f"vllm_text={choice.message.content!r}")
    print(f"vllm_logprob_tokens={len(choice.logprobs.content)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
