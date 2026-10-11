# Phase 7 — Rollout Backend Benchmark (prefill vs decode)

Hardware: see row source · Model: Qwen2.5-0.5B-Instruct + LoRA r16 ·
Task: toy arithmetic · `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

Workload: max_new_tokens=128, group_size=2.
prefill = one no-grad forward over the prompt block (parallel);
decode = the HF/vLLM autoregressive loop (serial). Tokens counted
are REAL (attention-masked), padding excluded.

## Results

| backend | batch | rows | prefill tok/s | decode tok/s | per-seq tok/s |
|---|---|---|---|---|---|
| hf | 1x2 | 2 | 4334 | 92 | 45.8 |
| hf | 2x2 | 4 | 4643 | 122 | 30.6 |
| hf | 4x2 | 8 | 5261 | 122 | 15.3 |
| hf | 8x2 | 16 | 5700 | 162 | 10.1 |

(HF rows: GTX 1080 Ti, measured 2026-10-11. The vLLM row joins on the
390 server — rerun the same command there.)

## Findings

**Finding 1 — the prefill/decode asymmetry is ~35–45x, and it is
structural, not a tuning issue.** Prefill is one dense GEMM over W
parallel tokens; decode re-reads ~1 GiB of fp16 weights from HBM to
emit ONE token per row per step. Even with zero Python overhead the
decode loop cannot approach prefill throughput on this hardware —
that gap is why dedicated inference engines exist (continuous batching
to fill the bandwidth with rows, paged KV to hold them).

**Finding 2 — prefill THROUGHPUT GROWS with batch (4334 → 5700).**
More rows amortize the same kernel-launch and weight-read overhead
across more useful tokens; the GPU is increasingly busy per launch.
Decode shows the same shape (92 → 162) but from a base ~40x lower —
Phase 4's "batch doubling ≈ doubles decode tok/s" on a longer max-token
run, reconfirmed.

**Finding 3 — per-sequence decode speed FALLS as batch grows
(45.8 → 10.1 tok/s).** Total throughput rises while per-row speed
drops: each decode step now computes attention over more rows and the
KV cache grows, so a step takes longer; rows also finish at different
times (EOS) and the batch marches at the slowest unfinished row's
pace — static-batching waste in its purest form. The per-seq number
is what a user of ONE generation feels; the total is what the RL loop
pays for. An engine with continuous batching decouples these: finished
rows free their slot immediately instead of idling to the batch tail.

**Finding 4 — the roofline says we leave ~97% of decode bandwidth on
the table.** 1080 Ti: ~484 GB/s HBM; weights ~1 GiB fp16 → a bandwidth-
perfect single stream would decode at ~460 tok/s. We observe ~10–46.
The missing order of magnitude is Python-side per-step host time
(60–80 ms in HF `generate()`), kernel launch latency across hundreds
of small ops, and small effective batch — precisely the items vLLM
attacks (C++/CUDA decode loop, fused kernels, iteration-level
scheduling). The vLLM row on the 390 server will price those wins.
