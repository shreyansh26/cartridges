# Cartridges

This repo exposes a single public benchmark entrypoint:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=3 python scripts/run_benchmark.py wikipedia_india
```

## India 1024 Run

Command used for the latest fixed explicit `1024`-token India run:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=3 python scripts/run_benchmark.py wikipedia_india \
  --gpu 3 \
  --device cuda:0 \
  --run-name india_1024_eos_fix \
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

- [india_1024_eos_fix](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_eos_fix)

Key output files from that run:

- Run manifest: [run_manifest.json](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_eos_fix/run_manifest.json)
- Shared baseline predictions: [predictions.jsonl](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_eos_fix/baseline/predictions.jsonl)
- Bootstrap questions: [questions.txt](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_eos_fix/bootstrap/questions.txt)
- Teacher answers: [teacher_answers.jsonl](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_eos_fix/bootstrap/teacher_answers.jsonl)
- Training dataset: [train_dataset.jsonl](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_eos_fix/bootstrap/train_dataset.jsonl)
- Cartridge checkpoint: [text-0_checkpoint.pt](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_eos_fix/cartridge_1024/train/text-0_checkpoint.pt)
- Cartridge artifact: [text-0_cartridge.pt](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_eos_fix/cartridge_1024/train/text-0_cartridge.pt)
- Cartridge predictions: [predictions.jsonl](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_eos_fix/cartridge_1024/predictions.jsonl)
- Cleaned cartridge predictions: [predictions_cleaned.jsonl](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_eos_fix/cartridge_1024/predictions_cleaned.jsonl)
- Per-budget summary: [summary.json](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_eos_fix/cartridge_1024/report/summary.json)
- Cleaned per-budget summary: [summary.json](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_eos_fix/cartridge_1024/report_cleaned/summary.json)
- Per-budget report: [comparison.md](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_eos_fix/cartridge_1024/report/comparison.md)
- Cleaned per-budget report: [comparison.md](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_eos_fix/cartridge_1024/report_cleaned/comparison.md)
- Aggregate run summary: [summary.json](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_eos_fix/report/summary.json)
- Aggregate run report: [comparison.md](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_eos_fix/report/comparison.md)
- vLLM log: [vllm.log](/mnt/ssd1/shreyansh/home_dir/cartridges/outputs/wikipedia_india/runs/india_1024_eos_fix/logs/vllm.log)

Observed summary for `cartridge_1024`:

- Baseline exact match: `0.90`
- Cartridge exact match: `0.75`
- Average compression ratio: `8.07x`
- Average prefill speedup: `8.00x`
- Average end-to-end speedup: `1.92x`
- One-time build time: `115.44s`
