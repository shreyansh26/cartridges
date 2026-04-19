from __future__ import annotations

"""Thin wrapper over the vLLM OpenAI-compatible API plus tokenizer parity checks."""

import os
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from openai import OpenAI
from transformers import AutoTokenizer

from cartridges.config import DEFAULT_MATRIX, TOKENIZER_PROBE_TEXT


@dataclass
class TopLogprobCandidate:
    """One candidate from a top-k token distribution."""
    token: str
    token_id: int | None
    logprob: float


@dataclass
class TokenLogprob:
    """Top-k logprob metadata for one generated token."""
    token: str
    token_id: int
    logprob: float
    top_logprobs: list[TopLogprobCandidate]
    source: Literal["vllm", "hf_teacher"]


@dataclass
class TokenizerParityReport:
    """Comparison of local and server-side tokenization for a fixed probe string."""
    model_id: str
    model_revision: str | None
    probe_text: str
    local_token_count: int
    local_token_ids: list[int]
    server_token_count: int
    server_token_ids: list[int]
    matches: bool


@dataclass
class ChatCompletionResult:
    """Normalized chat response returned by the vLLM client wrapper."""
    text: str
    token_ids: list[int]
    token_logprobs: list[TokenLogprob]
    raw_logprobs: list[dict[str, Any]]
    usage: dict[str, int] | None
    finish_reason: str | None
    logprob_source: Literal["vllm", "hf_teacher", "none"]


class VLLMClient:
    """Handle chat completions, tokenization probes, and teacher-logprob fallback."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "cartridges-local",
        model_id: str | None = None,
        model_revision: str | None = None,
        tokenizer_id: str | None = None,
        tokenizer_revision: str | None = None,
        teacher_device: str | None = None,
        timeout_seconds: float = 60.0,
    ):
        """Construct a vLLM client plus the matching local tokenizer."""
        self.base_url = base_url.rstrip("/")
        self.server_root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        self.api_key = api_key
        self.model_id = model_id or DEFAULT_MATRIX.model_id
        self.model_revision = model_revision or os.environ.get("CARTRIDGES_MODEL_REVISION")
        self.tokenizer_id = tokenizer_id or self.model_id
        self.tokenizer_revision = tokenizer_revision or os.environ.get(
            "CARTRIDGES_TOKENIZER_REVISION",
            self.model_revision,
        )
        self.teacher_device = teacher_device or os.environ.get("CARTRIDGES_HF_TEACHER_DEVICE")
        self.timeout_seconds = timeout_seconds
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.http_client = httpx.Client(timeout=self.timeout_seconds)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_id,
            revision=self.tokenizer_revision,
        )
        self._teacher_model = None

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self.http_client.close()

    def _auth_headers(self) -> dict[str, str]:
        """Build headers for direct HTTP endpoints outside the OpenAI SDK."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _tokenize_via_server(self, prompt: str) -> list[int]:
        """Tokenize text through vLLM's tokenizer endpoint for parity checks."""
        response = self.http_client.post(
            f"{self.server_root}/tokenize",
            headers=self._auth_headers(),
            json={"prompt": prompt},
        )
        response.raise_for_status()
        payload = response.json()
        tokens = payload.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
            raise RuntimeError(f"Unexpected /tokenize payload: {payload}")
        return tokens

    def _render_prompt_text(self, messages: list[dict[str, str]]) -> str:
        """Render messages with the local tokenizer's chat template."""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _best_effort_candidate_id(self, token_text: str) -> int | None:
        """Resolve token text back to an ID when it maps to exactly one token locally."""
        token_ids = self.tokenizer.encode(token_text, add_special_tokens=False)
        if len(token_ids) == 1:
            return token_ids[0]
        return None

    def _extract_completion_ids(
        self,
        messages: list[dict[str, str]],
        completion_text: str,
    ) -> list[int]:
        """Recover completion token IDs by differencing prompt-only and prompt+completion tokenizations."""
        if not completion_text:
            return []
        prompt_text = self._render_prompt_text(messages)
        prompt_ids = self._tokenize_via_server(prompt_text)
        combined_ids = self._tokenize_via_server(prompt_text + completion_text)
        if len(combined_ids) < len(prompt_ids):
            raise RuntimeError(
                "Combined prompt+completion tokenization is shorter than "
                "prompt tokenization."
            )
        return combined_ids[len(prompt_ids) :]

    def _vllm_logprobs_complete(
        self,
        raw_logprobs: list[dict[str, Any]],
        completion_token_ids: list[int],
        requested_top_logprobs: int,
    ) -> bool:
        """Check whether vLLM returned enough logprob data to avoid HF fallback."""
        if not completion_token_ids:
            return True
        if len(raw_logprobs) != len(completion_token_ids):
            return False
        return all(
            len(entry.get("top_logprobs", [])) >= requested_top_logprobs for entry in raw_logprobs
        )

    def _materialize_vllm_logprobs(
        self,
        raw_logprobs: list[dict[str, Any]],
        completion_token_ids: list[int],
    ) -> list[TokenLogprob]:
        """Convert raw vLLM logprob payloads into the repo's normalized structure."""
        token_logprobs: list[TokenLogprob] = []
        for entry, token_id in zip(raw_logprobs, completion_token_ids, strict=True):
            token_text = entry["token"]
            top_logprobs = [
                TopLogprobCandidate(
                    token=candidate["token"],
                    token_id=self._best_effort_candidate_id(candidate["token"]),
                    logprob=float(candidate["logprob"]),
                )
                for candidate in entry["top_logprobs"]
            ]
            token_logprobs.append(
                TokenLogprob(
                    token=token_text,
                    token_id=token_id,
                    logprob=float(entry["logprob"]),
                    top_logprobs=top_logprobs,
                    source="vllm",
                )
            )
        return token_logprobs

    def _load_teacher_model(self):
        """Lazily load the local HF teacher model used only for smoke-mode fallback."""
        if self._teacher_model is not None:
            return self._teacher_model
        import torch
        from transformers import AutoModelForCausalLM

        model_kwargs: dict[str, Any] = {
            "revision": self.model_revision,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        if self.teacher_device:
            model_kwargs["device_map"] = {"": self.teacher_device}
        self._teacher_model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        self._teacher_model.eval()
        return self._teacher_model

    def _normalize_usage(self, usage: Any) -> dict[str, int] | None:
        """Normalize OpenAI SDK usage objects into plain dictionaries."""
        if usage is None:
            return None
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        if not isinstance(usage, dict):
            return None
        normalized: dict[str, int] = {}
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(field)
            if isinstance(value, int):
                normalized[field] = value
        return normalized or None

    def _teacher_top_logprobs(
        self,
        messages: list[dict[str, str]],
        completion_text: str,
        top_logprobs: int,
    ) -> tuple[list[int], list[TokenLogprob]]:
        """Recompute top-k logprobs locally when vLLM does not provide complete ones."""
        import torch

        if not completion_text:
            return [], []

        model = self._load_teacher_model()
        prompt_text = self._render_prompt_text(messages)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        completion_token_ids = self.tokenizer.encode(completion_text, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids + completion_token_ids], device=model.device)

        with torch.inference_mode():
            logits = model(input_ids=input_ids).logits
        response_start = len(prompt_ids)
        response_end = response_start + len(completion_token_ids)
        response_logits = logits[:, response_start - 1 : response_end - 1, :]
        log_probs = torch.log_softmax(response_logits, dim=-1)
        target_ids = input_ids[:, response_start:response_end]
        target_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(0).squeeze(-1)
        top_values, top_indices = torch.topk(log_probs.squeeze(0), k=top_logprobs, dim=-1)

        token_logprobs: list[TokenLogprob] = []
        for row, token_id in enumerate(completion_token_ids):
            top_candidates = [
                TopLogprobCandidate(
                    token=self.tokenizer.decode([candidate_id]),
                    token_id=int(candidate_id),
                    logprob=float(candidate_logprob),
                )
                for candidate_id, candidate_logprob in zip(
                    top_indices[row].tolist(),
                    top_values[row].tolist(),
                    strict=True,
                )
            ]
            token_logprobs.append(
                TokenLogprob(
                    token=self.tokenizer.decode([token_id]),
                    token_id=token_id,
                    logprob=float(target_log_probs[row]),
                    top_logprobs=top_candidates,
                    source="hf_teacher",
                )
            )
        return completion_token_ids, token_logprobs

    def get_served_model(self) -> str:
        """Return the single model currently advertised by the vLLM server."""
        model_list = self.client.models.list()
        if not model_list.data:
            raise RuntimeError("vLLM server returned no models.")
        return model_list.data[0].id

    def assert_server_model_matches(self) -> None:
        """Ensure the server is actually serving the model this run expects."""
        served_model = self.get_served_model()
        if served_model.lower() != self.model_id.lower():
            raise RuntimeError(f"Expected server model {self.model_id}, got {served_model}.")

    def probe_tokenizer_parity(self) -> TokenizerParityReport:
        """Compare local tokenization against the server tokenizer on a fixed probe string."""
        local_ids = self.tokenizer.encode(TOKENIZER_PROBE_TEXT, add_special_tokens=False)
        if not local_ids:
            raise RuntimeError("Local tokenizer probe returned no token ids.")
        self.assert_server_model_matches()
        server_ids = self._tokenize_via_server(TOKENIZER_PROBE_TEXT)
        return TokenizerParityReport(
            model_id=self.model_id,
            model_revision=self.model_revision,
            probe_text=TOKENIZER_PROBE_TEXT,
            local_token_count=len(local_ids),
            local_token_ids=local_ids,
            server_token_count=len(server_ids),
            server_token_ids=server_ids,
            matches=local_ids == server_ids,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        max_completion_tokens: int,
        temperature: float = 0.0,
        top_logprobs: int | None = None,
        run_mode: Literal["smoke", "full"] = "smoke",
    ) -> ChatCompletionResult:
        """Run one chat completion and optionally collect top-k token supervision."""
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            logprobs=top_logprobs is not None,
            top_logprobs=top_logprobs,
        )
        choice = response.choices[0]
        raw_logprobs: list[dict[str, Any]] = []
        if choice.logprobs and choice.logprobs.content:
            for entry in choice.logprobs.content:
                raw_logprobs.append(
                    {
                        "token": entry.token,
                        "logprob": entry.logprob,
                        "top_logprobs": [
                            {
                                "token": candidate.token,
                                "logprob": candidate.logprob,
                            }
                            for candidate in (entry.top_logprobs or [])
                        ],
                    }
                )
        completion_text = choice.message.content or ""
        completion_token_ids = self._extract_completion_ids(messages, completion_text)
        token_logprobs: list[TokenLogprob] = []
        logprob_source: Literal["vllm", "hf_teacher", "none"] = "none"
        if top_logprobs is not None:
            # Full runs require server-side top-k parity; smoke runs may fall back to local HF
            # so the rest of the pipeline can still be exercised when vLLM logprobs are incomplete.
            if self._vllm_logprobs_complete(raw_logprobs, completion_token_ids, top_logprobs):
                token_logprobs = self._materialize_vllm_logprobs(raw_logprobs, completion_token_ids)
                logprob_source = "vllm"
            elif run_mode == "smoke":
                completion_token_ids, token_logprobs = self._teacher_top_logprobs(
                    messages=messages,
                    completion_text=completion_text,
                    top_logprobs=top_logprobs,
                )
                logprob_source = "hf_teacher"
            else:
                raise RuntimeError(
                    "vLLM returned incomplete top-logprobs; full mode is blocked until "
                    "server-side logprobs are complete."
                )
        return ChatCompletionResult(
            text=completion_text,
            token_ids=completion_token_ids,
            token_logprobs=token_logprobs,
            raw_logprobs=raw_logprobs,
            usage=self._normalize_usage(response.usage),
            finish_reason=choice.finish_reason,
            logprob_source=logprob_source,
        )
