# Phase 5 — Padding Benchmark

Hardware: GTX 1080 Ti 11GB · Model: Qwen2.5-0.5B-Instruct + LoRA r16
(fp32 adapters on frozen fp16 base) · Task: toy arithmetic ·
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

Workload: 5 steps per config, max_new_tokens=256, micro_batch_size=4.
`pad (batch)` is the response-block padding ratio of the FULL
batch (identical for both variants — bucketing changes what each
MICRO-BATCH pays, not the batch itself).

Regenerate with:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run -n tinyrl python scripts/benchmark_padding.py \
  --max-steps 5 --n-prompts 4,8 \
  --micro-batch-size 4
```

## Results

| batch | variant | pad (batch) | train s/step | train tok/s | peak (GiB) |
|---|---|---|---|---|---|
| 4×2 | fixed | 54% | 1.29 | 78 | 3.70 |
| 4×2 | bucketed | 55% | 1.17 | 103 | 3.70 |
| 8×2 | fixed | 71% | 3.91 | 86 | 6.42 |
| 8×2 | bucketed | 72% | 2.79 | 120 | 6.40 |

## Findings

**Finding 1 — the waste was real: 54–72% of the response block is
padding.** The batch pads every response to the longest one sampled;
toy-task responses range from ~8 tokens (`24 + 11 = 35`) to 40+,
so most rows carry mostly filler. Worse with wider batches: at 8×2 the
ratio hits 71% — the more rows ride together, the longer the tail that
sets the width. `pad (batch)` is identical across variants by
construction: it measures the batch BEFORE micro-batching.

**Finding 2 — bucketing + trim buys +32–40% training throughput.**
4×2: 78 → 103 tok/s; 8×2: 86 → 120 tok/s (s/step 3.91 → 2.79). The
mechanism: after sorting by length, each micro-batch's width is the
max of ITS OWN rows, not of the whole batch, so the `[n, T_bucket, V]`
logits (and their fp32 copies and backward graph) shrink toward the
real content. The gain GROWS with batch width — exactly where the
padding ratio is worst.

**Finding 3 — peak memory barely moves, and Phase 3 already told us
why.** The peak is set during `prepare()`'s full-batch no-grad logprob
passes (Phase 3, Finding 4 in `benchmarks/memory.md`) — those still run
on the un-bucketed batch, so their `[N, T, V]` transient sets the
floor. Micro-batch-level trimming cannot lower a peak the micro-batch
never defines. (Next lever, same as Phase 3's: chunk the logprob passes.)

**Finding 4 — correctness was free: gradients stay exactly equal to
the full batch.** Token-weighted accumulation (`scale = chunk tokens /
total tokens`) telescopes to the full-batch mean for ANY chunking —
`tests/test_bucketing.py` proves grads match on a ragged batch, and
the old `1/n_micro` scaling is documented there as biased. Sorting only
changes row order; group-relative advantages are per-row and
order-agnostic, so nothing upstream shifts.

**Finding 5 — what we did NOT fix.** Rollout still generates the whole
batch at the full padded width (left-padded prompts, HF `generate()`),
and left-side prompt padding is untouched (`rows(trim=)` only cuts the
right; trimming left would require re-shifting every row — RoPE makes
it harmless, so it stays). A token-budget micro-batcher (varying rows
per micro-batch to equalize tokens) is the natural next step; the
weighted accumulation already supports arbitrary chunk sizes.
Production systems solve the same problem from the other side —
continuous batching keeps only same-length groups in flight at all.
