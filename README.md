# Cartridges

This repo exposes a single public benchmark entrypoint:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=3 python scripts/run_benchmark.py wikipedia_india
```

## India 1024 Run

Command used for the latest explicit `1024`-token India run:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=3 python scripts/run_benchmark.py wikipedia_india \
  --gpu 3 \
  --device cuda:0 \
  --run-name india_1024_readme \
  --cartridge-tokens 1024 \
  --train-steps 240 \
  --bootstrap-count 120 \
  --max-completion-tokens 48 \
  --max-context-tokens 8192
```

Inputs used:

- Corpus: [data.txt](/mnt/ssd1/shreyansh/home_dir/cartridges/data/wikipedia_india/data.txt)
- Eval spec: [eval_spec.json](/mnt/ssd1/shreyansh/home_dir/cartridges/data/wikipedia_india/eval_spec.json)
- Metadata: [metadata.json](/mnt/ssd1/shreyansh/home_dir/cartridges/data/wikipedia_india/metadata.json)

Run output root:

- [india_1024_readme](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_readme)

Key output files from that run:

- Run manifest: [run_manifest.json](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_readme/run_manifest.json)
- Shared baseline predictions: [predictions.jsonl](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_readme/baseline/predictions.jsonl)
- Bootstrap questions: [questions.txt](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_readme/bootstrap/questions.txt)
- Teacher answers: [teacher_answers.jsonl](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_readme/bootstrap/teacher_answers.jsonl)
- Training dataset: [train_dataset.jsonl](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_readme/bootstrap/train_dataset.jsonl)
- Cartridge checkpoint: [text-0_checkpoint.pt](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_readme/cartridge_1024/train/text-0_checkpoint.pt)
- Cartridge artifact: [text-0_cartridge.pt](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_readme/cartridge_1024/train/text-0_cartridge.pt)
- Cartridge predictions: [predictions.jsonl](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_readme/cartridge_1024/predictions.jsonl)
- Per-budget summary: [summary.json](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_readme/cartridge_1024/report/summary.json)
- Per-budget report: [comparison.md](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_readme/cartridge_1024/report/comparison.md)
- Aggregate run summary: [summary.json](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_readme/report/summary.json)
- Aggregate run report: [comparison.md](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_readme/report/comparison.md)
- vLLM log: [vllm.log](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_readme/logs/vllm.log)

Observed summary for `cartridge_1024`:

- Baseline exact match: `0.90`
- Cartridge exact match: `0.20`
- Average compression ratio: `8.07x`
- Average prefill speedup: `8.32x`
- Average end-to-end speedup: `0.53x`
- One-time build time: `142.26s`
