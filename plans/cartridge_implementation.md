# Plan: Standalone Single-GPU Cartridges Reproduction

**Generated**: 2026-04-02

## Overview
- Build a clean `src/`-layout implementation in the current folder and keep `cartridges_ref/` read-only as reference only.
- Target a local reproducible proof first: NIAH-style long-context retrieval as the primary benchmark, plus an arXiv-text smoke path to verify generic corpus support.
- Use `Qwen/Qwen3-4B` as the parent model, `vllm serve` for synthesis and parent-model baseline quality runs, and local Hugging Face inference for cartridge training plus matched-engine performance measurement.
- Use offline W&B for smoke runs. Promote to online only when `WANDB_API_KEY` is present.
- On April 2, 2026, GPUs 3 and 5 were idle; execution should prefer GPU 3, then GPU 5, and never use GPUs 6 or 7.

## Current Public Interface
- Public scripts are now limited to `scripts/check_env.py`, `scripts/serve_vllm.py`, and `scripts/run_benchmark.py`.
- Versioned experiment inputs live under `data/<experiment_name>/`.
- Required files for each experiment are `data.txt` and `eval_spec.json`; `metadata.json` is optional.
- Benchmark outputs live under `outputs/<experiment_name>/runs/<run_id>/` with a `latest` symlink for convenience.
- The unified runner computes the full-context baseline once, then trains and evaluates one cartridge per requested budget under the same run directory.

## Prerequisites
- `uv` available for environment and dependency management
- Hugging Face auth via `HF_TOKEN`
- A free allowed GPU: prefer 3, fallback 5, never 6 or 7
- `WANDB_API_KEY` only for full online runs; smoke runs default to offline
- vLLM-compatible model serving on the same pinned model/tokenizer revision used by local HF

## Interfaces
- CLI entrypoints: `scripts/check_env.py`, `scripts/compat_check.py`, `scripts/serve_vllm.py`, `scripts/prepare_niah.py`, `scripts/prepare_arxiv_smoke.py`, `scripts/synthesize_self_study.py`, `scripts/train_cartridge.py`, `scripts/eval_baseline.py`, `scripts/eval_cartridge.py`, `scripts/run_local_repro.py`.
- `ConversationRecord`: corpus id, chunk provenance, messages, assistant token ids, assistant top-k token ids/logprobs, model/tokenizer revision.
- `CartridgeCheckpoint`: model/tokenizer revision, cartridge token budget, frozen-token count, per-layer K/V tensors, optimizer/scheduler/scaler/global-step/RNG state.
- `EvalRecord`: prompt id, method, prediction, gold, exact-match, canonical KV bytes, compression ratio, prefill ms, decode tok/s, total latency ms.

## Dependency Graph

```text
T1
├─ T2
├─ T3
├─ T4
T2 ─┬─ T5 ─┐
T3 ─┘      │
T4 ─┬─ T6 ─┼─ T8 ─ T9
    ├─ T7a┐│
T2 ─┘     ││
T3 ───────┘│
T6 ─────── T7b
T7a ──────┐
T7b ──────┴─ T7c ─┘
```

## Tasks

### T1: Bootstrap Environment And Compatibility
- **depends_on**: `[]`
- **location**: `pyproject.toml`, `.env.example`, `scripts/check_env.py`, `scripts/compat_check.py`, `src/cartridges/config.py`
- **description**: create the `uv` project scaffold; define train/infer/dev dependency groups; centralize environment resolution including `resolve_wandb_mode`; add hard checks for free port, allowed GPU availability, VRAM thresholds, HF auth, and W&B gating; pin one compatibility matrix before any implementation proceeds. Version policy: prefer the newest stable `transformers` 4.x release, the newest stable `vllm` release, and the matching supported `torch`; if `compat_check` fails, downgrade `transformers` within 4.x before changing `vllm`.
- **validation**: `uv run scripts/check_env.py --mode smoke` returns a hard failure for no allowed GPU or port conflict, a gated warning for missing `WANDB_API_KEY`, and a selected GPU of 3 or 5; `uv run scripts/compat_check.py` loads Qwen3-4B, runs one forward/backward step with cartridge params, and confirms vLLM chat plus logprobs support.
- **status**: Completed
- **log**: Created the `uv` project scaffold, `.venv`, compatibility matrix, and centralized W&B mode resolution. Implemented `scripts/check_env.py` with hard checks for HF auth, port conflicts, allowed GPUs, and smoke/full gating. Implemented `scripts/compat_check.py` and validated the local HF stack with `uv run --extra train --extra infer scripts/compat_check.py --skip-vllm-runtime --device cuda:3`. The workspace is not a git repository, so no commit was possible. vLLM runtime probing is implemented in the checker but deferred until T3 brings up the server path.
- **files edited/created**: `pyproject.toml`, `.env.example`, `src/cartridges/__init__.py`, `src/cartridges/config.py`, `scripts/check_env.py`, `scripts/compat_check.py`

### T2: Build Deterministic Benchmark And Corpus Prep
- **depends_on**: `[T1]`
- **location**: `src/cartridges/data/`, `scripts/prepare_niah.py`, `scripts/prepare_arxiv_smoke.py`
- **description**: implement deterministic NIAH-style corpus generation/loading and exact-match eval splits as the primary benchmark; add an arXiv-paper text smoke corpus; standardize manifests for corpus text, chunk metadata, eval queries, and split membership. Use stable row hashes and manifest checksums instead of promising byte-stable Parquet.
- **validation**: rerunning prep with the same seed produces identical row hashes and manifest checksums; benchmark prep hard-fails if any eval prompt would exceed the configured max length.
- **status**: Completed
- **log**: Implemented deterministic local NIAH data prep and arXiv smoke manifest generation with stable row hashes and location-independent manifest checksums. Fixed the initial NIAH prompt-budget bug by trimming context tokens until every query fits within `max_seq_length`. Validated with repeated `prepare_niah.py` runs in different output directories and confirmed identical row hashes and manifest hashes. The workspace is not a git repository, so no commit was possible.
- **files edited/created**: `src/cartridges/data/__init__.py`, `src/cartridges/data/common.py`, `src/cartridges/data/niah.py`, `src/cartridges/data/arxiv_smoke.py`, `scripts/prepare_niah.py`, `scripts/prepare_arxiv_smoke.py`

### T3: Build The vLLM Parent-Model Path
- **depends_on**: `[T1]`
- **location**: `src/cartridges/clients/vllm_openai.py`, `scripts/serve_vllm.py`
- **description**: wrap the OpenAI-compatible `vllm serve` API for chat generation, token/logprob extraction, and tokenizer-revision parity checks; pin a single model and tokenizer revision shared by vLLM and local HF; validate that assistant top-k logprobs are present. For smoke only, if vLLM logprobs are incomplete, fall back to local-HF teacher logits; full mode stays blocked until vLLM logprobs work.
- **validation**: a smoke request returns text, token ids, usage, and top-k logprobs; a fixed probe string tokenizes identically in both vLLM and local HF.
- **status**: Completed
- **log**: Extended the vLLM client to return structured completion token ids plus per-token top-logprob records, added tokenizer parity checks against the server `/tokenize` endpoint, and wired smoke-only local-HF teacher fallback when vLLM logprobs are incomplete while keeping full mode hard-blocked. Updated the vLLM launcher to expose model/tokenizer revision pinning, served model naming, tokenizer info endpoint enablement, and max-logprobs configuration so the server path uses the same pinned identifiers as local HF. Static validation passed via `compileall`, `ruff`, and an in-process mocked smoke check that verified tokenizer parity, structured token extraction, usage capture, and the full-mode block path. On April 3, 2026, live validation also passed against a real `vllm serve` instance on GPU 3 via `CUDA_VISIBLE_DEVICES=3 uv run --extra train --extra infer scripts/compat_check.py --device cuda:0 --base-url http://127.0.0.1:8000/v1 --api-key cartridges-local`, which returned chat logprobs successfully. The workspace is not a git repository, so no commit was possible.
- **files edited/created**: `src/cartridges/clients/vllm_openai.py`, `src/cartridges/clients/__init__.py`, `scripts/serve_vllm.py`

### T4: Implement The Cartridge Core
- **depends_on**: `[T1]`
- **location**: `src/cartridges/core/`, `src/cartridges/models/`, `tests/test_cartridge_cache.py`
- **description**: implement trainable per-layer KV tensors, initialization from the first `p` corpus tokens, frozen attention-sink handling, canonical KV-byte accounting, and checkpoint save/load. Primary integration path is standard Transformers cache injection with single-example batches. If the cache API cannot backpropagate cartridge params on Qwen3 during the validation test, automatically switch to a thin vendored Qwen3 attention wrapper in this repo and use that path for the rest of the implementation.
- **validation**: base-model grads remain zero while cartridge grads are non-zero; cartridge reload round-trips; logits from initialized cartridge match logits from the literal first-`p` prefix within tolerance on a fixed sample.
- **status**: Completed
- **log**: Added a trainable per-layer KV cartridge module with frozen-token support, canonical KV-byte accounting, save/load round-trips, and a `DynamicCache` wrapper compatible with current Transformers cache APIs. Strengthened validation with prefix-equivalence and frozen-base-model gradient tests on a tiny causal LM. The workspace is not a git repository, so no commit was possible.
- **files edited/created**: `src/cartridges/core/cartridge.py`, `src/cartridges/core/__init__.py`, `src/cartridges/models/__init__.py`, `tests/test_cartridge_cache.py`

### T5: Build SELF-STUDY Synthesis
- **depends_on**: `[T2, T3]`
- **location**: `src/cartridges/synthesis/`, `scripts/synthesize_self_study.py`
- **description**: implement chunk sampling, the five seed-prompt families from the paper, self-play conversation generation through vLLM, and a compact on-disk schema storing messages, assistant supervision, and provenance. Smoke mode uses a small sample count; full mode starts at 4k conversations and can scale to 16k if acceptance is not met.
- **validation**: smoke synthesis writes a valid dataset artifact with non-empty assistant top-k supervision and passes schema validation on every row.
- **status**: Completed
- **log**: Added a standalone SELF-STUDY synthesis path with the five paper seed-prompt families, corpus-slice loading for both NIAH and arXiv smoke manifests, compact conversation/supervision records, and a CLI that targets the live vLLM OpenAI-compatible server. Normalized vLLM usage metadata down to stable integer token counts so synthesis artifacts validate cleanly. Validated end-to-end with a real smoke run on GPU 3 using `scripts/synthesize_self_study.py` against `http://127.0.0.1:8000/v1`, producing `/tmp/cartridges_synth_smoke/conversations.jsonl` with non-empty assistant supervision for every row. The workspace is not a git repository, so no commit was possible.
- **files edited/created**: `src/cartridges/synthesis/__init__.py`, `src/cartridges/synthesis/self_study.py`, `scripts/synthesize_self_study.py`, `src/cartridges/clients/vllm_openai.py`

### T6: Build The Distillation Training Loop
- **depends_on**: `[T4, T5]`
- **location**: `src/cartridges/train/`, `scripts/train_cartridge.py`
- **description**: train only cartridge K/V parameters against stored assistant top-k distributions; freeze the parent model; use bf16, batch size 1, and gradient accumulation; support full-state checkpoint resume including optimizer, scheduler, scaler, global step, and RNG. Smoke mode uses 20 to 50 optimizer steps; full mode escalates step count before changing benchmarks.
- **validation**: a smoke run decreases KL or cross-entropy on a tiny synthesized set and a resume-equivalence test reproduces the same next-step loss within tolerance.
- **status**: Completed
- **log**: Added a standalone cartridge training loop that loads synthesized conversations, initializes a trainable KV cartridge from the system prompt prefix, computes a sparse distillation loss from stored top-logprob supervision, freezes the parent model, and saves both cartridge-only and full-state checkpoints with optimizer, scheduler, RNG, and step state. Validated on the smoke synthesis artifact at `/tmp/cartridges_synth_smoke/conversations.jsonl`: a 20-step run on GPU 3 reduced loss from `0.9500` to `0.2256`, and a split resume check (`10 + 10` resumed vs fresh `20`) matched within `9.49e-4` final-loss difference. The workspace is not a git repository, so no commit was possible.
- **files edited/created**: `src/cartridges/train/__init__.py`, `src/cartridges/train/cartridge.py`, `scripts/train_cartridge.py`

### T7a: Build Baseline Quality And Matched-Engine Performance
- **depends_on**: `[T2, T3, T4]`
- **location**: `src/cartridges/eval/`, `scripts/eval_baseline.py`
- **description**: implement two baseline paths: naive long-context quality evaluation through vLLM, and a matched-engine local-HF baseline for performance numbers. Quality claims may use vLLM; throughput and latency claims must use matched-engine local HF so the backend is identical to cartridge inference.
- **validation**: baseline evaluation emits the standard `EvalRecord` schema and includes canonical KV bytes plus matched-engine timing fields.
- **status**: Completed
- **log**: Added a shared `EvalRecord` schema, NIAH prompt/scoring helpers, a vLLM quality baseline path, and a matched-engine local-HF baseline path with explicit prefill/decode timing and canonical KV-byte accounting. Validated on smoke NIAH eval rows in two phases on GPU 3: first `scripts/eval_baseline.py --methods vllm_quality` while `vllm serve` was live, then after shutting the server down `scripts/eval_baseline.py --methods hf_matched --device cuda:0` under `CUDA_VISIBLE_DEVICES=3`. Both passes wrote schema-valid JSONL artifacts in `/tmp/cartridges_baseline_smoke`, and both sample rows scored exact-match successfully. The workspace is not a git repository, so no commit was possible.
- **files edited/created**: `src/cartridges/eval/__init__.py`, `src/cartridges/eval/common.py`, `src/cartridges/eval/baseline.py`, `scripts/eval_baseline.py`

### T7b: Build Cartridge Evaluation
- **depends_on**: `[T2, T4, T6]`
- **location**: `src/cartridges/eval/`, `scripts/eval_cartridge.py`
- **description**: run local-HF inference with a loaded cartridge, exact-match scoring on NIAH, canonical KV-byte accounting, and timing collection. Use the same prompt set and decoding settings as T7a’s matched-engine baseline.
- **validation**: cartridge evaluation emits the same `EvalRecord` schema as T7a and passes exact-match scoring on a fixed tiny fixture.
- **status**: Completed
- **log**: Added a cartridge-backed matched-engine evaluation path that loads a saved `TrainableKVCartridge`, runs local-HF greedy decoding with cartridge-prefilled KV cache, reports timing fields, and computes compression ratios against the naive full-context KV size. Validated in two ways on GPU 3: first, `/tmp/cartridge_eval_smoke.jsonl` showed schema-valid cartridge records on real NIAH eval rows; second, a fixed tiny fixture built from a matched-engine HF teacher on one NIAH row was distilled into `/tmp/cartridge_fixture_train/niah-0_cartridge.pt`, and `scripts/eval_cartridge.py` scored that fixture with exact-match `True` and compression ratio `16.06x`. The workspace is not a git repository, so no commit was possible.
- **files edited/created**: `src/cartridges/eval/cartridge.py`, `src/cartridges/eval/common.py`, `src/cartridges/eval/__init__.py`, `scripts/eval_cartridge.py`

### T7c: Merge And Summarize Comparison Results
- **depends_on**: `[T7a, T7b]`
- **location**: `src/cartridges/eval/reporting.py`, `reports/templates/`
- **description**: merge baseline and cartridge results, compute compression ratios from the canonical tensor-shape formula, and produce a comparison table plus a short markdown summary. Runtime GPU memory remains a secondary diagnostic only.
- **validation**: merged output contains paired rows for every prompt and method and reports quality, compression, and matched-engine throughput deltas.
- **status**: Completed
- **log**: Added a reporting module that pairs baseline and cartridge records by `prompt_id`, computes compression and throughput ratios, and writes both JSONL and markdown summaries. Validated by merging `/tmp/cartridges_baseline_smoke/baseline_hf_matched.jsonl` with `/tmp/cartridge_eval_smoke.jsonl` into `/tmp/cartridge_comparison_smoke`, which produced paired rows for both smoke prompts plus aggregate compression and throughput metrics. The workspace is not a git repository, so no commit was possible.
- **files edited/created**: `src/cartridges/eval/reporting.py`, `src/cartridges/eval/__init__.py`, `reports/templates/comparison_report.md`

### T8: Build The Single-GPU Orchestrator
- **depends_on**: `[T5, T6, T7c]`
- **location**: `scripts/run_local_repro.py`, `configs/*.yaml`
- **description**: create smoke and full configs plus one driver with explicit single-GPU phase boundaries: start vLLM for synthesis and vLLM quality baseline, stop and verify the server process exits, run local-HF training and matched-engine evals, then restart vLLM only if needed. Make runs idempotent with run-scoped output dirs, sentinel files, and `--resume-from`.
- **validation**: `uv run scripts/run_local_repro.py --mode smoke` completes end-to-end on one allowed GPU without overlapping vLLM and local-HF residency.
- **status**: Completed
- **log**: Added smoke/full YAML configs and a single-driver orchestrator that performs environment selection, deterministic NIAH prep, vLLM startup and health checks, synthesis, vLLM quality baseline, explicit vLLM shutdown, local-HF training, local-HF baseline eval, cartridge eval, and final report generation with per-phase sentinel files under `.done/`. Smoke validation passed end-to-end via `uv run --extra train --extra infer scripts/run_local_repro.py --mode smoke --run-name smoke_ci`, producing a complete run in `outputs/smoke_ci` with all expected sentinel files and no overlapping vLLM/local-HF residency. The workspace is not a git repository, so no commit was possible.
- **files edited/created**: `configs/smoke.yaml`, `configs/full.yaml`, `scripts/run_local_repro.py`

### T9: Execute The Full Reproduction Until Acceptance
- **depends_on**: `[T8]`
- **location**: `outputs/`, `reports/local_repro.md`
- **description**: run the full NIAH reproduction on GPU 3, fallback 5, with cartridge budgets `{512, 1024}` first; if acceptance is missed, escalate to `{512, 1024, 2048}` and/or 16k synthesis samples before changing anything else. Use the longest context that passes preflight on one H100, targeting 64k and falling back to 32k if needed. Keep smoke offline; use online W&B only if `WANDB_API_KEY` is present.
- **validation**: final artifacts include checkpoints, synthesis manifests, baseline and cartridge eval tables, and a report that shows at least one budget with `compression_ratio >= 4x`, `matched_engine_throughput_ratio >= 1.5x`, and exact-match within 5 absolute points of the naive baseline; if not, the sweep continues.
- **status**: Not Completed
- **log**: Added a held-out India Wikipedia benchmark harness on April 3, 2026 with an 8192-token frozen corpus, a 20-question final unseen eval set, passage-level bootstrap question generation, and strict single-GPU phase separation between vLLM teacher generation and local-HF supervision/training. Validated the sequencing fix with `/tmp/india_wikipedia_benchmark_smoke_seq`, then ran two real held-out comparisons on GPU 3: `/tmp/india_wikipedia_benchmark_main1` at a 512-token cartridge budget and `/tmp/india_wikipedia_benchmark_main2` at a 1024-token budget. The 512-token run reached `0.75` strict exact-match versus a `0.90` full-context baseline with `16.14x` compression, `8.39x` prefill speedup, and `2.14x` average end-to-end speedup; the 1024-token run reached `0.65` strict exact-match with `8.07x` compression and `2.17x` average end-to-end speedup. Decode-only throughput stayed near `1.0x`, which is expected because the speedup comes from prefill rather than decode. Acceptance is still open because no run has yet met the exact-match target while preserving the speedup target under the strict metric.
- **files edited/created**: `examples/india_wikipedia_8192.txt`, `examples/india_wikipedia_8192.metadata.json`, `examples/india_wikipedia_20_eval_spec.json`, `src/cartridges/benchmarks/__init__.py`, `src/cartridges/benchmarks/wikipedia_qa.py`, `scripts/run_india_wikipedia_benchmark.py`, `reports/india_wikipedia_8192_comparison.md`

## Parallel Execution Groups

| Wave | Tasks | Can Start When |
|------|-------|----------------|
| 1 | `T1` | Immediately |
| 2 | `T2`, `T3`, `T4` | `T1` complete |
| 3 | `T5` | `T2`, `T3` complete |
| 4 | `T6`, `T7a` | `T4`+`T5` for `T6`; `T2`+`T3`+`T4` for `T7a` |
| 5 | `T7b` | `T2`, `T4`, `T6` complete |
| 6 | `T7c` | `T7a`, `T7b` complete |
| 7 | `T8` | `T5`, `T6`, `T7c` complete |
| 8 | `T9` | `T8` complete |

## Testing Strategy
- Unit tests: cartridge gradient isolation, save/load round-trip, canonical KV-byte calculation, tokenizer-revision parity, NIAH scoring, W&B mode resolution.
- Integration tests: smoke vLLM serve plus logprobs, smoke synthesis, 20 to 50 step training, matched-engine baseline eval, cartridge eval, orchestrator resume after an injected failure.
- Acceptance: the first full report must prove lower canonical KV bytes and higher matched-engine throughput than naive full-context prompting, not just lower wall-clock time on a different backend.

## Risks & Mitigations
- **Transformers/vLLM/Qwen compatibility is fragile**: pin exact versions and enforce `scripts/compat_check.py` before implementation depends on the stack.
- **vLLM top-logprobs may be incomplete**: fail fast in full mode; allow local-HF teacher fallback only for smoke mode.
- **Single-GPU process overlap can OOM**: orchestrator must explicitly stop vLLM before HF training or cartridge eval starts.
- **Engine mismatch can distort performance claims**: use vLLM only for parent-model quality and synthesis; use matched-engine local HF for throughput and latency comparisons.
- **Parquet byte-stability is unrealistic**: validate determinism with stable row hashes and manifest checksums instead.
- **Missing `WANDB_API_KEY` blocks online runs**: centralize mode gating and require offline smoke runs by default.

## Assumptions And Defaults
- `cartridges_ref/` remains reference-only and is not modified.
- DeepWiki indexing for `HazyResearch/cartridges` was unavailable during planning, so the plan is based on local reference inspection plus the paper.
- v1 excludes Tokasaurus, SGLang, Modal, and `pydrantic`.
- v1 training uses single-example batches instead of packed multi-sequence flex attention.
- Full online W&B runs remain blocked until `WANDB_API_KEY` is set.

## Post-Plan Additions
- Refactored the public interface on April 3, 2026 to a single generic benchmark path:
  - `scripts/run_benchmark.py <experiment_name>` is now the only public benchmark entrypoint.
  - `data/wikipedia_india/` is the canonical checked-in experiment folder replacing `examples/`.
  - The old demo-specific, India-specific, NIAH-specific, and arXiv-specific public scripts were removed.
  - Generic dataset helpers now live in `src/cartridges/data/text_dataset.py`.
  - Generic benchmark helpers now live in `src/cartridges/benchmarks/text_benchmark.py`.
- Added a held-out India Wikipedia benchmark workflow on April 3, 2026:
  - `data/wikipedia_india/data.txt` freezes the India corpus at 8192 tokens.
  - `data/wikipedia_india/eval_spec.json` contains the 20 final unseen factual questions.
  - `scripts/run_benchmark.py wikipedia_india --cartridge-tokens 512 1024` runs the single-GPU comparison end to end and writes the shared baseline, per-budget predictions, and aggregate report under `outputs/wikipedia_india/runs/<run_id>/`.

## Sources
- [Cartridges paper](https://arxiv.org/abs/2506.06266)
- [HazyResearch/cartridges reference repo](https://github.com/HazyResearch/cartridges)
- [Transformers KV cache docs](https://huggingface.co/docs/transformers/main/en/kv_cache)
- [vLLM OpenAI-compatible server docs](https://github.com/vllm-project/vllm/blob/main/docs/serving/openai_compatible_server.md)
