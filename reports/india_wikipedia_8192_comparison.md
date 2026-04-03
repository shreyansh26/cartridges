# India Wikipedia 8192-Token Benchmark

## Setup

- Corpus: `examples/india_wikipedia_8192.txt`
- Corpus metadata: `examples/india_wikipedia_8192.metadata.json`
- Held-out evaluation set: `examples/india_wikipedia_20_eval_spec.json`
- Runner: `scripts/run_india_wikipedia_benchmark.py`
- Parent model: `Qwen/Qwen3-4B`
- Evaluation backend: matched-engine local HF for both full-context and cartridge inference
- Teacher-data phase: vLLM only
- Cartridge build phase: local HF only after vLLM shutdown

## Commands

```bash
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=3 python scripts/run_india_wikipedia_benchmark.py \
  --gpu 3 \
  --device cuda:0 \
  --work-dir /tmp/india_wikipedia_benchmark_main1 \
  --bootstrap-count 120 \
  --cartridge-tokens 512 \
  --train-steps 240 \
  --max-completion-tokens 48

CUDA_VISIBLE_DEVICES=3 python scripts/run_india_wikipedia_benchmark.py \
  --gpu 3 \
  --device cuda:0 \
  --work-dir /tmp/india_wikipedia_benchmark_main2 \
  --bootstrap-count 120 \
  --cartridge-tokens 1024 \
  --train-steps 240 \
  --max-completion-tokens 48
```

## Results

| Budget | Baseline EM | Cartridge EM | Compression | Decode Throughput | Prefill Speedup | End-to-End Speedup | Build Time | Baseline Follow-up | Cartridge Follow-up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 tokens | 0.90 | 0.75 | 16.14x | 0.98x | 8.39x | 2.14x | 138.77s | 393.44 ms | 189.41 ms |
| 1024 tokens | 0.90 | 0.65 | 8.07x | 1.01x | 8.28x | 2.17x | 144.47s | 401.10 ms | 189.60 ms |

## Interpretation

- The cartridge path is cheaper on every query in total latency. The speedup is in prefill, not decode. Decode throughput stays near `1.0x` because both methods decode with the same local HF model.
- The one-time cost is cartridge construction, not the first question after construction. In both runs, the first cartridge query is already faster than the first full-context query.
- With a `512`-token cartridge, the strict exact-match gap is `15` points against the full-context baseline, but the latency and compression gains are strong.
- With a `1024`-token cartridge, strict exact match is worse than `512` because several misses are formatting or answer-order artifacts, not obvious retrieval failures.

## Strict-EM Misses

### 512-token cartridge

- `india-q08`: gold `bangladesh and myanmar`; prediction `Bangladesh and Myanmar.`
- `india-q09`: gold `sri lanka and the maldives`; prediction `Sri Lanka and the Maldives.`
- `india-q12`: gold `buddhism and jainism`; prediction `Hinduism and Jainism`
- `india-q16`: gold `361`; prediction `342`
- `india-q18`: gold `4`; prediction `2`

### 1024-token cartridge

- `india-q03`: gold `seventh-largest`; prediction `7th-largest`
- `india-q08`: gold `bangladesh and myanmar`; prediction `Bangladesh and Myanmar.`
- `india-q09`: gold `sri lanka and the maldives`; prediction `Sri Lanka and the Maldives.`
- `india-q10`: gold `indus river basin`; prediction `Indus`
- `india-q11`: gold `indus valley civilisation`; prediction `Indus Valley Civilisation.`
- `india-q12`: gold `buddhism and jainism`; prediction `Jainism and Buddhism`
- `india-q20`: gold `8`; prediction `1`

## Raw Outputs

- `512`-token run:
  - `/tmp/india_wikipedia_benchmark_main1/report/comparison.md`
  - `/tmp/india_wikipedia_benchmark_main1/report/summary.json`
- `1024`-token run:
  - `/tmp/india_wikipedia_benchmark_main2/report/comparison.md`
  - `/tmp/india_wikipedia_benchmark_main2/report/summary.json`
