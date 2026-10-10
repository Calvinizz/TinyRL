# Phase 4 — Throughput Benchmark

Hardware: GTX 1080 Ti 11GB · Model: Qwen2.5-0.5B-Instruct + LoRA r16
(fp32 adapters on frozen fp16 base) · Task: toy arithmetic ·
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

Workload: 5 steps per config, max_new_tokens=256, sweep over batch width.

Regenerate with:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run -n tinyrl python scripts/benchmark_throughput.py \
  --max-steps 5 --n-prompts 4,8
```

## Results — section time (mean s/step)

| batch | rollout | reward | prepare | train | Σ measured |
|---|---|---|---|---|---|
| 4×2 | 1.15 | 0.00 | 0.26 | 1.35 | 2.75 |
| 8×2 | 1.44 | 0.00 | 0.51 | 2.82 | 4.77 |

## Results — throughput

| batch | gen tok/s | train tok/s | gen : train |
|---|---|---|---|
| 4×2 | 102 | 87 | 1x |
| 8×2 | 198 | 101 | 1x |

## Findings

**Finding 1 — where the step time goes.** Rollout and the training
pass are co-dominant (4×2: 1.15 s vs 1.35 s; 8×2: 1.44 s vs 2.82 s —
training pulls ahead as batch grows). `prepare` (old+ref logprob
passes) sits at ~10%, `reward` is invisible at 0.00 s — it is CPU regex
over a handful of strings; it will never matter at this scale. The
optimizer step is folded into "train" and costs nothing measurable:
AdamW over 2.16 M LoRA params is a few small kernels, dwarfed by the
`[N, T, V]` logits forward.

**Finding 2 — the headline: generation scales with batch, training
doesn't.** Doubling the batch (4×2 → 8×2) almost doubles generation
throughput (102 → 198 tok/s, ×1.94) while rollout *time* only rises
×1.26 (1.15 → 1.44 s). Decode is memory-bandwidth-bound: every decode
step re-reads the ~1 GB of weights from VRAM no matter how many
sequences ride along, so more sequences = more tokens per weight-read,
nearly free. Training throughput moves only ×1.16 (87 → 101): the
forward+backward is already parallel over all tokens, so its cost grows
with the token count it consumes.

**Finding 3 — per-sequence decode speed is flat (~12-13 tok/s) and
that flatness is the whole serving business.** 102/8 ≈ 12.8 and
198/16 ≈ 12.4 tok/s per sequence — adding sequences leaves each one
just as fast (until compute saturates). The flip side: a single
sequence can never go faster than one decode step, which here is
~60-80 ms — HF `generate()` spends most of that in Python-level
per-step overhead (sampling loop, logits processing), not in the
matmul. This is exactly the gap dedicated rollout engines (vLLM,
SGLang) attack — and why Phase 7 abstracts the backend. (The naive
"decode is 100× slower than training" story is about *per-sequence*
decode vs one parallel forward; batched decode hides it at the batch
level, which is why gen ≈ train tok/s in the table.)

**Finding 4 — padding is the unmeasured tax.** Every logged token
count excludes padding, but the training forward computes logits for
padded positions too — T is set by the *longest* response in the
batch. The Phase-3 memory finding (`[N, T, V]` dominates) and this
Phase-4 compute waste are the same elephant: Phase 5 measures the
padding ratio and buckets by length to stop paying it.

**Finding 5 — per-step peak, now visible, ties the phases together.**
With the per-step high-water reset, the step log shows peak climbing
with response length (4.90 → 6.51 → 9.87 GiB across three 4×2 steps at
max_new_tokens=384, one long response away from the OOM fallback).
Memory pressure here is not a fixed cost — it is a function of the
length distribution the policy itself samples.
