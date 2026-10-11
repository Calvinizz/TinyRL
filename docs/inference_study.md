# Phase 7 Study — Why RL Systems Use Dedicated Rollout Engines

Companion to `tinyr/rollout/` (backend abstraction) and
`benchmarks/rollout.md`. Every number quoted here is TinyRL's own
measurement unless marked otherwise; the server-measured numbers land
in the marked sections after the 390 runs.

## 1. Prefill vs decode — the 100x asymmetry

Autoregressive generation has two regimes:

- **Prefill**: one parallel forward over the prompt. All W tokens are
  known, so attention is a single `[W, W]` matrix multiply — a dense
  GEMM, the shape GPUs are best at.
- **Decode**: one token per forward step. Each step re-reads ALL model
  weights (494M params, fp16 ≈ 1 GiB) from HBM to produce ONE token —
  so a single stream is bound by **memory bandwidth**, not FLOPs.

TinyRL receipts: on the 1080 Ti, generation sits at ~12 tok/s per
sequence (Phase 4) while training's teacher-forced forwards chew the
same tokens 100+ tok/s. And doubling the batch nearly doubles decode
throughput (102 → 198 tok/s, Phase 4) — the smoking gun that decode is
bandwidth-bound: the weight re-read is AMORTIZED across rows, so more
rows ride the same GiB of weights for nearly free.

The roofline arithmetic (1080 Ti: ~484 GB/s, model ~1 GiB fp16):
12 tok/s × 1 GiB ≈ 12 GB/s of weight traffic — i.e. we use ~2.5% of
the bandwidth ceiling. The other 97% is lost to: Python-side sampling
loop (60–80 ms/step of HF `generate()` host time), kernel launch
latency (hundreds of tiny kernels per step), and small effective batch.

## 2. KV cache — the memory bill that grows with context

Without a KV cache, generating token t re-computes K/V for all t−1
prior tokens: O(t²) total. The cache stores each layer's K and V once:
per token,

```text
KV/token = 2 (K and V) × layers × kv_heads × head_dim × bytes
Qwen2.5-0.5B: 2 × 24 × 2 × 64 × 2 B (fp16) = 12 KiB/token
```

(GQA matters here: 2 KV heads instead of 14 attention heads — a 7x
cache cut vs. full MHA, which is exactly why small models ship GQA.)

Our bill is tiny: a 384-token rollout costs ~4.5 MiB per sequence, so
at batch 8 the cache is ~36 MiB next to the multi-GiB logits transient
(Phase 3's real villain). Scale the same math to a 70B model and the
KV cache DOMINATES: ~320 KiB/token (fp16, grouped) → a 32k context is
~10 GiB per sequence. That inversion — cache > weights — is what
forced the industry to engine-level solutions:

## 3. Continuous batching (iteration-level scheduling)

Static batching (what HF `generate()` does, and TinyRL today): the
batch is fixed; when the shortest sequence hits EOS, its slot idles
until the LONGEST finishes. With response lengths spanning 8 to 40+
tokens on our toy task (Phase 5 measured 54–72% padding), the tail of
every batch is mostly idle compute.

Continuous batching (Orca, 2021): the scheduler consults the queue
EVERY decode step — finished sequences exit immediately, waiting
requests join mid-flight. Batch composition changes per step; padding
waste collapses toward zero without any length bucketing (Phase 5's
`iter_micro_batches_bucketed` is the training-side cousin of the same
idea).

## 4. PagedAttention (vLLM)

The KV cache's second problem is FRAGMENTATION: pre-allocating a
max-length contiguous block per sequence wastes the unused tail
(internal fragmentation); variable-length requests churn the allocator
(external fragmentation). vLLM borrows OS paging: KV lives in fixed
blocks (16 tokens), logical → physical mapping per sequence, blocks
shared on prefix (prefix caching: our system prompt + chat template is
the same prefix EVERY rollout — free after the first batch).

Waste drops from ~60% (static pre-allocation) to <4% (one partial
block per sequence); the same paged layout enables copy-on-write
prefix sharing and fast beam search.

## 5. Why RL systems care (tying it together)

A GRPO loop is a generation farm wearing a training hat:

1. **Rollout is the wall-clock majority** — Phase 4: rollout dominates
   every step; training tok/s was never the bottleneck.
2. **The two phases want different software** — decode wants an
   inference engine (paged KV, continuous batching, no Python per
   token); training wants autograd + optimizer. No single runtime is
   good at both → `RolloutBackend` exists so TinyRL can host each on
   its best runtime.
3. **They want different hardware** — Phase 6: on one GPU the phases
   serialize on the same CUDA stream and decoupling buys nothing
   (46 tok/s both modes). The 390 server is the rematch: rollout on
   GPU 0 (vLLM), training on GPU 1, the Phase-6 queue in between.
4. **Weights move, data moves** — the moment rollout and training are
   separate processes/devices, every optimizer step must reach the
   engine (§7) and every experience must reach the trainer (our
   `ExperienceQueue` + version stamps, Phase 6).

## 6. vLLM measurements *(fill on the 390 server)*

- [ ] vLLM vs HF decode tok/s, same prompts, batch 1/4/8
  (`scripts/benchmark_rollout.py` adds the row automatically when
  vllm is importable)
- [ ] Per-sequence decode speed vs HF's ~12 tok/s ceiling
- [ ] TTFT vs batch (prefill queueing effect)
- [ ] KV-cache utilization report from the engine
- [ ] Phase 6 rematch: async across two GPUs vs one-GPU tie

## 7. Weight synchronization — P7.3 notes *(design; implement on server)*

The trainer's LoRA updates every step; the engine serves stale weights
until told. Three strategies, cheapest first:

1. **vLLM LoRA hot-swap** (`enable_lora=True`): after each optimizer
   step, save the adapter to a VERSIONED path and pass a
   `LoRARequest(name=f"policy_v{version}", path=...)` per generate
   call. Tiny payload (2.2M params ≈ 9 MiB), engine applies adapters
   per-request. Caveats: adapters are cached by NAME (version the
   name or they collide), and the sampled-from version must be
   stamped into the Experience — this plugs directly into Phase 6's
   `policy_version` / staleness gate, which was built for exactly
   this.
2. **collective_rpc weight push** (verl-style): execute a function
   inside the engine's worker process that copies a fresh state_dict
   into the runner's model. Exact full weights (not adapter
   arithmetic), one code path for any architecture — but a full-model
   transfer (~1 GiB fp16) per sync unless engine and trainer share a
   device group. This is what Phase 9's "find weight synchronization
   in verl" maps to.
3. **Merge-and-reload**: merge LoRA into the base, rebuild the engine.
   Correct but ~seconds of downtime per step — the strawman that
   shows why 1 and 2 exist.

TinyRL order: strategy 1 for P7.3 (smallest delta to the Phase-6
stamping machinery), study 2's code path during Phase 9.
