# TinyRL

**A miniature GRPO post-training framework for small LLMs — small enough to read in an afternoon, real enough to touch every systems problem that production RL frameworks solve.**

TinyRL trains Qwen2.5-0.5B-Instruct with GRPO (group-relative advantages, PPO clipping, k3 KL regularization) through a fully modular pipeline: pluggable rollout backends, an experience queue with staleness control, a vLLM engine with live LoRA weight hot-swap, and DDP trainers — all measured, all tested, all documented.

It was built the way the best systems are understood: one phase at a time, from a single-GPU minimal trainer to a multi-GPU engine + trainer world, with a benchmark receipt attached to every design decision.

---

## Headline results

| what | result | receipt |
|---|---|---|
| Peak GPU memory (11 GB card) | 7.63 → **3.99 GiB** (−48%) via micro-batching | `benchmarks/memory.md` |
| Training throughput (ragged responses) | **+32–40%** tok/s via length-bucketed micro-batches | `benchmarks/padding.md` |
| Decode speed (same RTX 3090) | HF `generate()` → vLLM: **13.4x** | `benchmarks/rollout.md` |
| Full training loop (async + vLLM + live weight sync) | lockstep 8.93 → **0.71 s/step** (12.6x) | `benchmarks/rematch.md` |
| DDP grad sync correctness | DDP grad == manual AllReduce average, **rel_err 6.4e-08**; replicas bitwise-identical | `scripts/ddp_minimal.py`, `docs/distributed_study.md` |
| 2-rank DDP world under a live engine | steady **0.70 s/step** = single-trainer parity, 0 drops | `benchmarks/ddp.md` |

---

## How it works

```text
            ┌──────────────────────────────────────────────┐
            │ RolloutWorker                                │
            │   RolloutBackend: HFRolloutBackend           │
            │                  | VLLMRolloutBackend        │
            └───────────────┬──────────────────────────────┘
                            │ prompts, batch of G responses each
                            ▼
            ┌──────────────────────────────────────────────┐
            │ RewardManager (rule-based, exact-answer)     │
            └───────────────┬──────────────────────────────┘
                            │ stamped with the policy version it sampled from
                            ▼
            ┌──────────────────────────────────────────────┐
            │ ExperienceQueue (bounded, blocking)          │
            │   depth = backpressure, staleness gate caps  │
            │   off-policy drift                           │
            └───────────────┬──────────────────────────────┘
                            ▼
            ┌──────────────────────────────────────────────┐
            │ GRPOTrainer                                  │
            │   group advantages · ratio + PPO clip        │
            │   k3 KL vs reference · micro-batching        │
            │   length-bucketed dynamic batching           │
            │   (optional DDP: prompt-level shards)        │
            └───────────────┬──────────────────────────────┘
                            │ LoRA adapter, versioned
                            ▼
            ┌──────────────────────────────────────────────┐
            │ sync_policy(): hot-swap into the vLLM engine │
            │ (LoRARequest per version — no restart)       │
            └──────────────────────────────────────────────┘
```

The distributed topology (`scripts/train_ddp.py`): a vLLM engine on its
own GPU, a torchrun world of DDP ranks beside it, joined by a
process-crossing queue. Rank 0 reads each experience and broadcasts it
(numpy payloads — torch tensors do not survive unrelated process
trees); every rank runs the identical staleness verdict, trains on a
prompt-level shard (groups are never split: advantages are intra-group
statistics), and rank 0 ships its post-AllReduce LoRA to the engine.

---

## Features

**Algorithm**
- GRPO: group-relative advantage `(rᵢ − mean)/std`, importance ratio, PPO-style clipping, k3 KL penalty against the reference policy
- LoRA (r16/α32 on qkvo) over an fp16 frozen base — the enabler for 11 GB GPUs
- Inner-epoch reuse of experiences with exact importance correction

**Memory**
- Micro-batch gradient accumulation (accumulated grads == full-batch grads, proven by test)
- Reference-model CPU offload with try/finally movement discipline
- OOM diagnostics + automatic group-size fallback (shrink batch, drain, retry)
- Live peak-memory profiling per step

**Throughput**
- Injectable-clock latency breakdown (rollout / reward / logprob passes / train)
- Length-bucketed dynamic batching with token-weighted accumulation — exact on ragged batches where naive `1/n` weighting is biased

**Async system design**
- Bounded blocking experience queue (depth = backpressure)
- Thread-safe policy versioning; every experience stamped with the version it was sampled from
- Staleness gate: drop experiences sampled more than N versions behind

**Inference**
- `RolloutBackend` interface; trainer is backend-agnostic
- HF `generate()` backend and vLLM backend behind one contract (byte-parity verified)
- Live LoRA weight sync into a running vLLM engine via versioned adapters

**Distributed**
- DDP with verified semantics: gradient AllReduce, replica-identity induction, `no_sync()` local gradients
- Exact global token-mean loss under DDP averaging (`ddp_loss_scale`, one scalar allreduce per step)
- Prompt-level sharding that never splits a GRPO group

---

## Quickstart

```bash
conda create -n tinyrl python=3.11 -y
conda activate tinyrl

# pick the torch build for your GPU (cu118 keeps Pascal/sm_61 kernels)
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft datasets accelerate

# download the base model (hf-mirror for CN networks; models/ is gitignored)
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct

# 1) minimal single-GPU GRPO on the toy arithmetic task
python scripts/train_tinyrl.py --toy --max-steps 10

# 2) same trainer, length-bucketed dynamic batching
python scripts/train_tinyrl.py --toy --max-steps 10 --dynamic-batching

# 3) producer/consumer decoupling with staleness control
python scripts/train_async.py --toy --mode async --max-steps 10

# 4) + vLLM engine with live weight sync (needs compute capability >= 7.0)
python scripts/train_async.py --toy --mode async --rollout-backend vllm --max-steps 10

# 5) the full distributed world (N GPUs; engine on GPU 0, DDP ranks on the rest)
python scripts/train_ddp.py --max-steps 10 --world 2 --trainer-gpus 1,2

# tests (GPU tests self-skip where hardware is missing)
pytest tests/
```

Any `TrainingConfig` field is a CLI flag (`--micro-batch-size 2`,
`--queue-size 4`, `--max-staleness 1`, ...).

---

## Repository layout

```text
tinyr/
├── config.py            # every knob in one dataclass
├── experience/          # Experience, buffer, queue, length metrics
├── models/              # ModelManager: policy + frozen reference
├── reward/              # rule-based RewardManager
├── rollout/             # RolloutWorker + backend interface (HF / vLLM)
├── trainer/             # GRPOTrainer, policy versioning, distributed helpers
├── profiling/           # MemoryProfiler, injectable-clock Timer
└── utils/               # shape/dtype/device assertions

scripts/                 # entry points: train_tinyrl / train_async / train_ddp
                         # ddp_minimal (the 3-claims DDP verification)
                         # benchmark_* (one per study)
benchmarks/              # results, one markdown per question asked
docs/                    # phase notes + inference & distributed studies
tests/                   # 109 unit tests (111 on the GPU server)
```

---

## Documentation

| doc | what's in it |
|---|---|
| `docs/phase1_results.md` | the first correct GRPO step, fp16/LoRA war stories |
| `docs/inference_study.md` | prefill vs decode, KV cache, PagedAttention — why RL systems use dedicated rollout engines |
| `docs/distributed_study.md` | DDP contract, replica-identity induction, ring AllReduce arithmetic, ZeRO ledger, DDP × RL |
| `benchmarks/memory.md` | the `[N, T, V]` logits tensor dominates this model class; why grad checkpointing is a no-op under LoRA |
| `benchmarks/throughput.md` | decode amortizes weight reads; per-sequence speed is host-bound on HF `generate()` |
| `benchmarks/padding.md` | 54–72% of the response block was padding; bucketing to the rescue |
| `benchmarks/decoupling.md` | single-GPU async is a wash — both phases serialize on one CUDA stream |
| `benchmarks/rematch.md` | the hardware rematch: decoupling pays on a 3090; the bottleneck flips to the trainer |
| `benchmarks/ddp.md` | engine + DDP ranks in one world; the IPC law (boundaries carry bytes, not tensors) |

---

## Field notes

Facts learned the hard way, all reproducible from this repo:

- The `[N, T, V]` logits tensor dwarfs everything else at 0.5B scale — micro-batch it and gradient checkpointing becomes a no-op under LoRA.
- Decode is host-bound: a 3090 ran HF decode *slower* than a 1080 Ti until the decode loop moved into vLLM.
- Async on one GPU buys nothing (same CUDA stream); the win requires separate silicon — which is why verl/OpenRLHF run dedicated rollout engines.
- fp16 has a dynamic-range tightrope: `logits.mean()` underflows gradients to exact zero (per-logit grad 5e-8 < min subnormal 6e-8); `logits.sum()` overflows past 65504. Bounded per-token gradients are the fix; GradScaler exists for the general case.
- LoRA makes DDP communication ~100x cheaper than full fine-tuning (8.7 MB vs 1.0 GB per step) — the mirror image of cheap weight sync.
- torch tensors cannot cross process trees: custom pickle reducers detour through shared-memory fd passing whose authkey challenge fails across unrelated processes. Boundaries carry bytes.

---

## Hardware

- **Baseline (Phases 1–6):** GTX 1080 Ti 11 GB, fp16 + LoRA. Everything except the vLLM backend runs here.
- **Full stack (Phases 7–8):** any machine with multiple CUDA GPUs (developed on 3 × RTX 3090). vLLM requires compute capability ≥ 7.0.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is recommended on small cards.

---

## Scope

TinyRL intentionally does **not**:

- implement CUDA kernels or NCCL
- target 70B-scale models or multi-node training
- support every post-training algorithm — it is GRPO, on purpose

> The focus is deliberately narrow: **GRPO + small LLM + measurable system optimization.**

---

## Status

Feature-complete through distributed training (2026-10). 109 unit tests green locally, 111 on the GPU server; every phase closed with an end-to-end GPU run and a benchmark receipt.

## License

TBD
