# Phase 6 — Sync vs Async Decoupling Benchmark

Hardware: GTX 1080 Ti 11GB · Model: Qwen2.5-0.5B-Instruct + LoRA r16 ·
Task: toy arithmetic · `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

Workload: 8 steps, batch 4x2, max_new_tokens=256, micro_batch_size=4,
queue_size=2, max_staleness=3 (the third row varies both, to force the
staleness gate to fire).
Both modes share one code path (queue -> stamp -> gate -> train),
same seed, model reloaded per run.

Regenerate:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run -n tinyrl python scripts/benchmark_decoupling.py \
  --max-steps 8 --n-prompts-per-step 4 --micro-batch-size 4
```

## Results

| mode | wall (s) | end-to-end tok/s | steps trained | drops | lag mean | lag max | peak (GiB) |
|---|---|---|---|---|---|---|---|
| sync | 22.1 | 46 | 8 | 0 | 0.00 | 0 | 4.28 |
| async | 33.3 | 46 | 8 | 0 | 1.12 | 2 | 5.91 |
| async, queue 6 / cap 1 | 36.8 | 40 | 7 | 1 | 1.00 | 2 | 5.91 |

## Findings

**Finding 1 — wall time says async is 51% slower; tokens/s says it is
free. Which is right?** The async run sampled 50% more response tokens
(1525 vs 1019) in its 33.3 s: same questions and seed, but the producer
in async samples from *different weights* (versions drift, and the
consumer's optimizer can step mid-generate — the prototype hazard), so
the sampled response lengths diverge. Normalized per token, throughput
is 46.1 vs 45.8 tok/s — a tie. The honest comparison metric for RL
pipelines is tokens/s, not steps/s: steps carry a random token payload.

**Finding 2 — on ONE GPU, decoupling buys ~nothing. That is the
lesson.** Rollout decode and training kernels both saturate the same
device, and ops from the two Python threads land on the same default
CUDA stream — they *serialize*, so "overlap" only hides Python-side
latency (the 60–80 ms/step of `generate()` bookkeeping from Phase 4),
which is small next to kernel time. Producer/consumer pays when the two
phases live on *different* hardware: a dedicated inference engine
(vLLM-style rollout pool) + training GPUs, which is exactly how verl /
OpenRLHF structure it. Our toy shows WHY that architecture exists.

**Finding 3 — async is not free even at equal throughput: +38% peak
memory and off-policy drift.** Peak 4.28 → 5.91 GiB: both phases
allocate concurrently (rollout KV-cache while training's `[N,T,V]`
logits pass is live), so the allocator sees their *sum* overlapping.
And lag mean 1.12: every second experience was sampled one optimizer
step stale — importance ratios still fix it in the loss (that is what
`ratio = exp(logp_new − logp_old)` is FOR), but clipping starts eating
updates as lag grows. Sync-mode lag is 0.00 *by construction*, the
only free lunch in this table.

**Finding 4 — the queue depth is the primary staleness bound; the
explicit cap is the second line of defense.** At queue_size=2 the max
lag observed was exactly 2 — backpressure alone caps how far the
producer can run ahead, so `max_staleness=3` never fired. Row 3
widens the queue to 6 and tightens the cap to 1: the gate then fires
(`drop | lag 2 > max_staleness 1 | sampled v5 | current v7`) and drops
1 of 8 experiences. Two redundant bounds is deliberate: depth controls
how *far ahead* the producer gets (memory + lag), the cap controls what
the trainer is *willing to eat* (algorithmic correctness) — with many
producers, or a fast producer on separate hardware, only the cap
protects you.

**Finding 5 — what a real system would add.** Atomic weight sync (the
consumer's in-place LoRA update can straddle one rollout across two
versions; our stamp is a lower bound on that rollout's true lag), a
rollout replica so generation never reads weights mid-update, and
experience replay prioritized by staleness instead of hard drops.
TinyRL keeps the drop — smallest correct valve — and documents the
rest.
