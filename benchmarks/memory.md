# Phase 3 — Memory Benchmark

Hardware: GTX 1080 Ti 11GB · Model: Qwen2.5-0.5B-Instruct + LoRA r16
(fp32 adapters on frozen fp16 base) · Task: toy arithmetic · batch 4×2 ·
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

Regenerate with:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run -n tinyrl python scripts/benchmark_memory.py --max-steps 5
```

## Results

(filled in by `scripts/benchmark_memory.py` after TODOs P3.1–P3.4)

## Findings

To fill after the runs:

- Which flag bought the most peak memory? Why?
- What did each flag cost in step time?
- Did `everything` stack the savings, or do they overlap?
- Reference numbers from Phase 1: full-FT fp16 + AdamW did not fit at
  all; LoRA with fp32 adapters was the enabler (see docs/phase1_results.md).
