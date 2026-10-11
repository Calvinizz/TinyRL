# Phase 8 Study — DDP, NCCL, ZeRO/FSDP

Companion to `scripts/ddp_minimal.py` (the hands-on) and
`tinyr/trainer/distributed.py` (the pure helpers). Every number is
TinyRL's own arithmetic on Qwen2.5-0.5B (494M params, our qkvo-LoRA
r16 = 2.16M trainable) unless marked otherwise.

## 1. The DDP contract

Data parallelism is one sentence with three clauses:

1. **Replicate** the model on every rank (same seed, same construction
   order → bitwise-identical start).
2. **Shard the data** — each rank trains on ITS slice (`shard_for_rank`,
   strided so any batch length stays balanced).
3. **AllReduce the gradients** — divide by world size, and every rank
   steps its own optimizer.

Everything else is implementation detail. Notably what is NOT in the
contract: a master process, weight broadcasting after the step, or any
rank being in charge. The average IS the agreement.

## 2. Why replicas never drift (the invariant chain)

Induction on optimizer steps:

```text
same weights_t  +  same grad_t (AllReduce)  +  same optimizer state_t
              → same weights_{t+1}
```

- **same grad**: DDP fires the AllReduce from hooks DURING backward —
  by the time `optimizer.step()` runs, every rank's `.grad` is already
  the world average. That is WHERE gradients merge: in backward, not
  in the optimizer.
- **same optimizer state**: AdamW is deterministic given identical
  inputs, and every rank steps the same number of times — so m, v
  evolve identically. Each rank's optimizer is fully in charge OF ITS
  OWN REPLICA, and that is enough.

Break any link and the replicas fork SILENTLY (no error, weights just
diverge over steps):

- rank-local RNG reaching the loss (dropout) → different losses →
  different "gradients of the same objective". This is why
  `ddp_minimal.py` builds its LoRA with `lora_dropout=0.0`.
- unequal shard sizes with a per-rank MEAN loss → mean-of-means ≠
  global mean (small shards get upweighted). `even_len` exists for the
  hang (below) AND for this bias.
- one rank skipping a backward → AllReduce mismatch → the world HANGS
  at a bucket that never arrives (the loudest failure mode in DDP:
  deadlock, not corruption).

## 3. What one step costs in bytes — NCCL and the ring

The AllReduce moves the gradient tensor once around a ring:

```text
bytes per GPU = 2(N-1)/N × grad_bytes     (N = world size)
N=2: ×1.0   N=3: ×1.33   N=4: ×1.5        (asymptote ×2)
```

implemented as reduce-scatter (every rank ends with 1/N of the SUM)
followed by all-gather (everyone collects everyone's 1/N). A ring
because every link carries the same V/N per step — bandwidth-optimal
and latency-tolerant, at the cost of 2(N-1) latency steps.

TinyRL's bill, per optimizer step:

| what syncs | bytes | over ~10 GB/s effective PCIe | note |
|---|---|---|---|
| our LoRA grads (fp32) | 2.16M × 4 B = **8.7 MB** | ~1 ms | invisible |
| full-model grads fp16 | 494M × 2 B = **1.0 GB** | ~100 ms | 100x |
| full-model grads fp32 | **2.0 GB** | ~200 ms | comm-bound |

**LoRA makes DDP nearly communication-free** — the adapter is 0.4% of
the parameters, so full-finetune data parallelism pays ~100x more
comm per step than ours at the same world size. The mirror image of
Phase 7's finding that LoRA makes weight SYNC cheap (9 MiB per push).

When does comm become THE bottleneck? When
`grad_bytes / per-step-FLOPs` grows: bigger model (more bytes),
smaller batch (fewer FLOPs between allreduces — tiny per-rank batches
make every step mostly comm). Rule of thumb at our scale: LoRA grads
never; 0.5B full-FT at N=2 already spends ~10% of a 1 s step in the
ring.

## 4. Bucketing — hiding comm under compute

DDP does not allreduce tensor-by-tensor (that would serialize on
latency). Parameters are grouped into buckets (default 25 MB);
backward fills them in reverse order of forward, and the moment a
bucket is FULL, its allreduce launches ASYNC while backward keeps
computing the next bucket. Comm overlaps compute for free; the last
bucket's allreduce is the only exposed tail. With our 8.7 MB of LoRA
grads this is one partial bucket — the machinery matters at full-FT
scale, not ours. (`no_sync()` — used in the minimal script to obtain
LOCAL gradients — is the same machinery with the launch suppressed:
the sanctioned idiom for gradient accumulation.)

## 5. The memory ledger — DDP vs ZeRO 1/2/3 vs FSDP

Canonical mixed-precision full fine-tune, per parameter:
fp16 weight 2 B + fp16 grad 2 B + fp32 master 4 B + fp32 Adam m 4 B
+ fp32 Adam v 4 B = **16 B/param**. Qwen2.5-0.5B → 7.9 GB of STATE,
plus activations.

DDP replicates all of it on every rank. The ZeRO ladder shards it:

| layout | replicated | sharded (/N) | N=2 | N=3 |
|---|---|---|---|---|
| DDP (=ZeRO-0) | — | everything: 7.9 GB | 7.9 | 7.9 |
| ZeRO-1 | params+grads 3.0 GB | optimizer 5.0 GB | 5.5 | 4.6 |
| ZeRO-2 | params 1.0 GB | grads+optimizer 6.9 GB | 4.4 | 3.3 |
| ZeRO-3 / FSDP | — | everything: 7.9 GB (+transient allgather) | 4.0 | 2.6 |

- **ZeRO-1**: optimizer states are pure function of grads — shard them,
  reduce-scatter grads to their owners, owners step their shard,
  all-gather the new weights. Same math, 1/N the optimizer memory.
- **ZeRO-2** adds gradient sharding (reduce-scatter means each rank
  only ever HOLDS 1/N of the grads — it was going to scatter them
  anyway).
- **ZeRO-3 / FSDP**: shard parameters too; all-gather each layer's
  flat shard on the way INTO forward, free it after. Now forward
  itself generates comm. FSDP is PyTorch-native ZeRO-3 with per-layer
  flat parameters — same ledger.
- What NO layout shards for free: **activations**. DDP cuts them 1/N
  anyway (each rank runs N/world rows — our 9.87 GiB trainer peak is
  logits-dominated, so 2-rank DDP halves exactly the thing that hurts).

TinyRL's actual ledger: frozen fp16 base 0.99 GB + LoRA fp32 8.7 MB +
grads 8.7 MB + Adam 17 MB ≈ **1.02 GB of state** — ZeRO has nothing
to save us; activations (Phase 3's [N,T,V] logits) are the whole game,
and plain DDP already shards those. ZeRO matters the day we drop LoRA.

## 6. DDP × RL — the axes SL never has

Supervised training parallelizes ONE axis (the gradient). RL has three:

1. **Trainer data-parallelism** — this phase; ranks split the
   experience batch.
2. **Rollout data-parallelism** — independent sampling engines; trivial
   for untrained correctness (no gradients), needs only prompt
   sharding + weight sync fanout.
3. **Weight synchronization** — the axis with no SL analogue: whatever
   the ranks agree the weights are, the ENGINE must be told, and told
   HONESTLY (Phase 7 §7: stamp `served_version`, let the staleness
   gate judge).

This is verl's actor/rolloff split; mapping TinyRL onto it is Phase 9.

## 7. P8.1 → P8.2 plan

P8.1 (`scripts/ddp_minimal.py`, current): the three claims above,
verified by hand on synthetic data — TODOs are (a) process-group init,
(b) the DDP wrap, (c) the proofs.

P8.2 (once green): engine + DDP trainers on the 3-GPU box. Two viable
topologies, decision deferred to scaffold time:

- **rank-0 dual role**: torchrun world of N; rank 0 hosts the vLLM
  engine AND trains (24 GiB fits engine at 0.35 + trainer ~10);
  experiences broadcast from rank 0; sync_weights runs on rank 0
  post-step (rank 0's replica IS the agreed average).
- **sidecar engine**: engine outside torchrun, talking to the training
  world through a process-crossing queue (mp.Queue). Cleaner roles,
  one more moving part.

Either way the Phase-6 stamps and staleness gate carry over unchanged
— they were built process-agnostic.
