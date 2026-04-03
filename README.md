# Cartridges

This repo exposes a single public benchmark entrypoint:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=3 python scripts/run_benchmark.py wikipedia_india
```

The benchmark reads:

- [data.txt](/mnt/ssd1/shreyansh/home_dir/cartridges/data/wikipedia_india/data.txt)
- [eval_spec.json](/mnt/ssd1/shreyansh/home_dir/cartridges/data/wikipedia_india/eval_spec.json)
- optional [metadata.json](/mnt/ssd1/shreyansh/home_dir/cartridges/data/wikipedia_india/metadata.json)

## India 1024 Stable Run

Command used:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=3 python scripts/run_benchmark.py wikipedia_india \
  --gpu 3 \
  --device cuda:0 \
  --run-name india_1024_stable \
  --cartridge-tokens 1024 \
  --train-steps 240 \
  --bootstrap-count 120 \
  --max-completion-tokens 48 \
  --max-context-tokens 8192 \
  --semantic-judge
```

Run output root:

- [india_1024_stable](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_stable)

Key output files:

- Run manifest: [run_manifest.json](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_stable/run_manifest.json)
- Baseline predictions: [predictions.jsonl](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_stable/baseline/predictions.jsonl)
- Bootstrap questions: [questions.txt](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_stable/bootstrap/questions.txt)
- Teacher answers: [teacher_answers.jsonl](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_stable/bootstrap/teacher_answers.jsonl)
- Training dataset: [train_dataset.jsonl](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_stable/bootstrap/train_dataset.jsonl)
- Cartridge checkpoint: [text-0_checkpoint.pt](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_stable/cartridge_1024/train/text-0_checkpoint.pt)
- Cartridge artifact: [text-0_cartridge.pt](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_stable/cartridge_1024/train/text-0_cartridge.pt)
- Training summary: [text-0_summary.json](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_stable/cartridge_1024/train/text-0_summary.json)
- Cartridge predictions: [predictions.jsonl](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_stable/cartridge_1024/predictions.jsonl)
- Per-budget summary: [summary.json](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_stable/cartridge_1024/report/summary.json)
- Per-budget report: [comparison.md](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_stable/cartridge_1024/report/comparison.md)
- Per-budget row data: [comparison.jsonl](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_stable/cartridge_1024/report/comparison.jsonl)
- Aggregate run summary: [summary.json](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_stable/report/summary.json)
- Aggregate run report: [comparison.md](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_stable/report/comparison.md)
- vLLM log: [vllm.log](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_stable/logs/vllm.log)

Observed summary for `cartridge_1024`:

- Baseline exact match: `0.90`
- Cartridge exact match: `0.55`
- Baseline semantic match: `1.00`
- Cartridge semantic match: `1.00`
- Average compression ratio: `8.07x`
- Average prefill speedup: `8.19x`
- Average end-to-end speedup: `1.73x`
- Baseline follow-up latency: `400.84 ms`
- Cartridge follow-up latency: `238.40 ms`
- One-time build time: `125.35 s`

## India 512 Stable Run

Command used:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=3 python scripts/run_benchmark.py wikipedia_india \
  --gpu 3 \
  --device cuda:0 \
  --run-name india_512_stable \
  --cartridge-tokens 512 \
  --train-steps 240 \
  --bootstrap-count 120 \
  --max-completion-tokens 48 \
  --max-context-tokens 8192 \
  --semantic-judge
```

Run output root:

- [india_512_stable](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_512_stable)

Key output files:

- Run manifest: [run_manifest.json](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_512_stable/run_manifest.json)
- Baseline predictions: [predictions.jsonl](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_512_stable/baseline/predictions.jsonl)
- Cartridge checkpoint: [text-0_checkpoint.pt](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_512_stable/cartridge_512/train/text-0_checkpoint.pt)
- Cartridge artifact: [text-0_cartridge.pt](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_512_stable/cartridge_512/train/text-0_cartridge.pt)
- Training summary: [text-0_summary.json](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_512_stable/cartridge_512/train/text-0_summary.json)
- Cartridge predictions: [predictions.jsonl](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_512_stable/cartridge_512/predictions.jsonl)
- Per-budget summary: [summary.json](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_512_stable/cartridge_512/report/summary.json)
- Per-budget report: [comparison.md](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_512_stable/cartridge_512/report/comparison.md)
- Per-budget row data: [comparison.jsonl](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_512_stable/cartridge_512/report/comparison.jsonl)
- Aggregate run summary: [summary.json](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_512_stable/report/summary.json)
- Aggregate run report: [comparison.md](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_512_stable/report/comparison.md)

Observed summary for `cartridge_512`:

- Baseline exact match: `0.90`
- Cartridge exact match: `0.45`
- Baseline semantic match: `1.00`
- Cartridge semantic match: `0.80`
- Average compression ratio: `16.14x`
- Average prefill speedup: `8.28x`
- Average end-to-end speedup: `1.76x`
- Baseline follow-up latency: `394.76 ms`
- Cartridge follow-up latency: `268.13 ms`
- One-time build time: `121.36 s`

Sanity check for the `512` run:

- Empty cartridge predictions: `0`
- Average completion tokens: `12.8`
- Max completion tokens: `33`
