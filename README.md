# TinyRL

TinyRL is a lightweight reinforcement-learning post-training framework for learning, experimenting with, and profiling LLM RL systems on limited hardware.

The project starts from a minimal **GRPO** implementation and gradually evolves toward a modular RL training engine with rollout/training decoupling, memory optimization, throughput profiling, and inference-backend abstraction.

> Goal: build the smallest RL post-training system that is **correct, measurable, modular, and explainable**.

---

## Overview

TinyRL is designed as a project-driven learning framework for understanding modern LLM post-training systems.

The initial focus is:

- GRPO training
- Single-GPU execution
- Small language models
- Limited-memory optimization
- RL training-system internals
- Progressive comparison with frameworks such as `verl`, `OpenRLHF`, and `TRL`

The project intentionally avoids large-scale distributed complexity at the beginning.

---

## Target Hardware

Initial development target:

- NVIDIA GTX 1080 Ti 11GB
- FP16
- Small LLMs: 0.5B ~ 1.5B
- Single GPU first

Recommended first experiment:

```text
Model: Qwen2.5-0.5B-Instruct
Task: GSM8K subset
Precision: FP16
Training: LoRA
Group Size: 2~4
```

---

## Architecture

Target data flow:

```text
Prompt Dataset
      ↓
Rollout Worker
      ↓
Reward Manager
      ↓
Experience
      ↓
Experience Buffer
      ↓
GRPO Trainer
      ↓
Policy Update
```

Later versions will further decouple rollout and training:

```text
        ┌───────────────┐
        │ RolloutWorker │
        └───────┬───────┘
                │
                ▼
        Experience Queue
                │
                ▼
        ┌───────────────┐
        │    Trainer    │
        └───────────────┘
```

---

## Project Structure

Planned structure:

```text
TinyRL/
├── tinyr/
│   ├── rollout/
│   ├── trainer/
│   ├── reward/
│   ├── experience/
│   ├── models/
│   └── utils/
├── configs/
├── scripts/
├── tests/
├── examples/
├── benchmarks/
├── docs/
└── README.md
```

---

## Roadmap

### Phase 0 — Environment & Project Setup

- [x] Create Python environment
- [x] Install PyTorch with CUDA support
- [x] Verify CUDA availability
- [x] Install Transformers
- [x] Install PEFT
- [x] Install Datasets
- [x] Install Accelerate
- [x] Install Weights & Biases
- [x] Verify `Qwen2.5-0.5B-Instruct` inference
- [x] Create the initial project structure

CUDA verification:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
```

Verified environment (2026-10-09):

```text
Python 3.11 (conda env: tinyrl)
torch 2.7.1+cu118          # cu118 is the last CUDA build with Pascal (sm_61) kernels
transformers 5.19.0 / peft 0.21.2 / datasets 5.1.0 / accelerate 1.15.0 / wandb 0.30.0
GPU: GTX 1080 Ti (sm_6.1), 10.9 GiB
```

Setup commands:

```bash
conda create -n tinyrl python=3.11 -y
conda activate tinyrl
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft datasets accelerate wandb

# download models into ./models (ignored by git)
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct

python scripts/verify_env.py
```

---

### Phase 1 — Minimal GRPO Trainer

Goal: implement one complete and correct GRPO optimization step.

#### RL Foundations

- [x] Policy Gradient
- [x] REINFORCE
- [x] Actor-Critic
- [x] GAE
- [x] PPO
- [x] GRPO

#### LLM RL Concepts

- [x] Understand prompt tokens vs. response tokens
- [x] Understand token-level log probabilities
- [x] Understand sequence-level rewards
- [x] Understand response masking
- [x] Understand the old policy
- [x] Understand the reference policy
- [x] Understand why old policy != reference policy
- [x] Understand KL regularization in LLM RL

#### Implementation

- [x] Load `Qwen2.5-0.5B-Instruct`
- [x] Load tokenizer
- [x] Prepare a small prompt dataset
- [x] Implement rollout generation
- [x] Generate multiple responses per prompt
- [x] Implement group sampling
- [x] Implement a rule-based reward
- [x] Start with GSM8K exact-answer reward
- [x] Implement group mean and standard deviation
- [x] Implement group-relative advantage

```text
A_i = (r_i - mean(r)) / (std(r) + eps)
```

- [x] Compute response-token log probabilities
- [x] Compute old-policy log probabilities
- [x] Compute reference-policy log probabilities
- [x] Implement the importance ratio

```text
ratio = exp(logp_new - logp_old)
```

- [x] Implement PPO-style clipping
- [x] Implement GRPO policy loss
- [x] Implement KL penalty
- [x] Mask prompt tokens from policy loss
- [x] Run one optimizer step successfully
- [x] Verify gradients are non-zero
- [x] Verify learning on a toy task

Verified runs and debugging notes: `docs/phase1_results.md` (trainer: `scripts/train_grpo.py`).

---

### Phase 2 — Modular Training Engine

Goal: move from a single training script to a small RL training system.

- [x] Implement `RolloutWorker`
- [x] Implement `RewardManager`
- [x] Implement `Experience`
- [x] Implement `ExperienceBuffer`
- [x] Implement `GRPOTrainer`
- [x] Implement `ModelManager`
- [x] Separate rollout logic from training logic

Planned experience structure:

```text
prompt_ids
response_ids
attention_mask
response_mask
rewards
advantages
old_logprobs
ref_logprobs
```

Engineering tasks:

- [x] Add shape assertions
- [x] Add dtype assertions
- [x] Add device assertions
- [x] Add unit tests for core components

Verified: 37 unit tests green + 10-step GPU toy run through the full
modular pipeline (`scripts/train_tinyrl.py`, `tinyr/` package).

---

### Phase 3 — Memory Optimization

Goal: make GRPO practical on an 11GB GPU.

#### Learn

- [x] Parameter memory
- [x] Gradient memory
- [x] Optimizer-state memory
- [x] Activation memory
- [x] KV-cache memory
- [x] FP32 vs. FP16 memory usage
- [x] Gradient accumulation
- [x] Activation checkpointing
- [x] CPU offload

#### Implement

- [x] FP16 training *(Phase 1: fp16 base + fp32 LoRA adapters)*
- [x] LoRA / PEFT support *(Phase 1: r16/α32 on qkvo)*
- [x] Gradient accumulation *(P3.3: `(loss / n_micro).backward()` per micro-batch; micro-accumulated grads == full-batch grads, `tests/test_accumulation.py`)*
- [x] Gradient checkpointing *(P3.2: +`enable_input_require_grads()`; bit-identical metrics verified — saves ~0 memory under qkvo-LoRA, see benchmark)*
- [x] Configurable micro-batch size *(P3.3: `iter_micro_batches` + `cfg.micro_batch_size`)*
- [x] Reference-model CPU offload *(P3.4: ref on CPU, moved to GPU per `prepare()` under try/finally)*
- [x] Peak GPU-memory profiling *(P3.1: `MemoryProfiler`)*
- [x] OOM diagnostics *(P3.5)*
- [x] Group-size fallback for low-memory hardware *(P3.5: shrink `n_prompts_per_step`, drain buffer, retry)*

#### Benchmark

- [x] Record baseline peak memory *(7.63 GiB @ 6×2 / 384 tok)*
- [x] Record optimized peak memory *(3.99 GiB with micro-batch 2 — `scripts/benchmark_memory.py`)*
- [x] Compare full fine-tuning vs. LoRA if feasible *(Phase 1 finding: full-FT fp16 + AdamW does not fit 11GB; LoRA with fp32 adapters is the enabler — see `docs/phase1_results.md`)*
- [x] Document results in `benchmarks/memory.md`

Verified: 45 unit tests green; five config variants produce bit-identical
KL/ratio metrics; OOM fallback recovers under stress (10.49 GB peak caught,
batch shrunk 12→11, training completed). Headline result: peak memory
7.63 → 3.99 GiB (−48%), and the benchmark shows *why* — the `[N, T, V]`
logits tensor dominates this model class, so micro-batching wins and
gradient checkpointing is a no-op under LoRA (`benchmarks/memory.md`).

---

### Phase 4 — Throughput Profiling

Goal: know where the step time goes.

- [x] Step-latency breakdown: rollout / reward / logprob passes / train *(P4.1: injectable-clock `Timer`; optimizer folded into "train" — a few kernels over 2M LoRA params, unmeasurable next to the `[N,T,V]` logits forward)*
- [x] Rollout tokens/s and training tokens/s *(P4.2)*
- [x] Peak GPU memory logged per step *(P4.2: per-step `reset_peak_memory_stats`)*
- [x] Record results in `benchmarks/throughput.md`

Verified: 52 unit tests green; per-step log now carries peak/GiB, gen
and train tok/s, and a section breakdown. Headline result
(`benchmarks/throughput.md`): doubling the batch nearly doubles
generation throughput (102 → 198 tok/s) while training throughput is
flat — decode amortizes weight reads, per-sequence decode speed is
stuck at ~12 tok/s on HF `generate()`, which is the case for dedicated
rollout engines (Phase 7).

---

### Phase 5 — Dynamic Batching & Padding Reduction

Goal: stop paying compute for padding tokens.

- [x] Measure padding ratio and the response-length distribution *(P5.1: `tinyr/experience/metrics.py`; pad% in the step log — 54–72% of the response block on the toy task)*
- [x] Bucket samples by response length *(P5.2: `iter_micro_batches_bucketed` + `Experience.rows(trim=)` cuts each micro-batch to its own longest real response)*
- [x] Dynamic micro-batching driven by the buckets *(P5.3: `cfg.dynamic_batching` branch in `train_step` — token-weighted accumulation; grads exactly == full-batch on ragged batches, `tests/test_bucketing.py`)*
- [x] Compare fixed vs dynamic batching throughput *(+32–40% train tok/s, `benchmarks/padding.md`)*

Verified: 64 unit tests green (incl. weighted-accumulation == full-batch
on a ragged batch, with the old `1/n_micro` bias documented as a
counterexample test); GPU run with `--dynamic-batching` produces sane
metrics and lower peaks. Headline: response-block padding was 54–72%;
bucketing shrinks each micro-batch to its own width for +32–40%
training throughput. Peak unchanged — it is set by the full-batch
no-grad logprob passes (Phase 3 Finding 4 again).

---

### Phase 6 — Rollout / Training Decoupling

Goal: the core RL-system design pattern — producer/consumer with
staleness control.

- [x] Experience queue between rollout and trainer *(P6.1: `ExperienceQueue` — bounded blocking hand-off built from deque + Condition; depth = backpressure)*
- [x] Rollout worker produces experiences; trainer consumes them *(producer thread / consumer loop in `scripts/train_async.py`, `--mode sync|async`)*
- [x] Policy version stamped on every experience *(P6.2: `PolicyVersion` thread-safe counter; `Experience.policy_version` stamped before `generate()` and carried through `rows()`)*
- [x] Staleness: log version lag, cap it with a max-staleness threshold *(P6.3 `policy_lag` + P6.4 consumer gate: drop experiences sampled more than `cfg.max_staleness` versions behind)*
- [x] Sync baseline vs simple async prototype: throughput comparison *(`scripts/benchmark_decoupling.py` → `benchmarks/decoupling.md`)*

Verified: 80 unit tests green (queue blocking/FIFO/stress, lockless-bump
race caught by a 4-thread test, stamp survives `rows()`); both modes run
on GPU with sane metrics — sync lag 0.00 by construction, async lag
mean 1.12 / max 2 at queue depth 2, and the staleness gate demonstrably
drops when the queue is widened and the cap tightened (queue 6 / cap 1
→ 1 drop of 8). Headline (`benchmarks/decoupling.md`): per-token
throughput is a tie (46 tok/s both modes) — on a single GPU the two
phases serialize on the same CUDA stream, so decoupling costs +38% peak
memory and off-policy drift while buying nothing; the win requires
rollout and training on separate devices, which is precisely why
verl/OpenRLHF run dedicated rollout engines.

---

### Phase 7 — Inference Backend Abstraction

Goal: the trainer shouldn't know how generation happens.

- [x] `RolloutBackend` interface (`generate(questions) -> RolloutResult`) *(framework: `tinyr/rollout/backend.py`; `RolloutWorker` is now a thin delegator)*
- [x] Hugging Face `generate()` backend behind it; trainer stays backend-agnostic *(P7.1: `tinyr/rollout/hf_backend.py`, GPU-smoke bit-identical through delegation)*
- [x] Study: prefill vs decode, KV cache, continuous batching, PagedAttention — why RL systems use dedicated rollout engines *(docs/inference_study.md; `benchmarks/rollout.md`: 35–45x asymmetry, plus the 390-server receipt — 3090 HF decode SLOWER than 1080 Ti, decode is host-bound)*
- [x] vLLM backend *(P7.2, 390 server: `tinyr/rollout/vllm_backend.py` — offline template + TokensPrompt, n=G, left/right-pad reconstruction, EOS-in-token_ids verified against HF; **13.4x decode** over HF on the same 3090; tests self-skip without vllm, 89/89 green on server)*
- [x] Weight sync: LoRA hot-swap with versioned adapters, stamped into Phase 6's `policy_version` *(P7.3: `sync_policy` in `tinyr/rollout/vllm_backend.py` — `LoRARequest(f"policy_v{v}", v, path)`, the int id IS the engine's adapter-cache key; `sync_weights` in `scripts/train_async.py` saves to `lora_cache/v{N}`; producer stamps `backend.served_version` so engine lag is honest lag. Acceptance: zero-adapter identity (fresh `lora_B==0` is token-identical greedy) + big-perturbation probes, `tests/test_backend.py::TestVLLMWeightSync`. First full-chain run — async + vLLM + live weights, 5 steps on one 3090: lag mean 1.80 max 3, 0 drops, KL 0→0.012 = weights visibly flowing)*

> Phase 7 verified end-to-end on the 390 server (2026-10-11): same-silicon
> decode 92 → 1235 tok/s (13.4x), contract byte-parity with the HF
> backend, weights flowing every optimizer step, and the Phase-6 rematch
> run (`benchmarks/rematch.md`): full chain at 0.71 s/step vs 8.93 s
> lockstep — 12.6x — with the bottleneck flipped to the trainer (rollout
> now runs ~2.4 steps ahead), which is exactly what Phase 8 attacks.

> 2026-10-11 un-deferred: vLLM requires compute capability >= 7.0, which
> the GTX 1080 Ti (sm_61) lacks — but Phase 7/8 now run on a 390 server
> (3 × RTX 3090, sm_86, driver 555 / CUDA 12.5), so real vLLM
> integration and the Phase-6 rematch (rollout on GPU 0, training on
> GPU 1) are back in scope. The 1080 Ti stays as the constrained-hardware
> baseline for every earlier benchmark.

---

### Phase 8 — Distributed Training Fundamentals

Goal: the vocabulary needed to read production RL frameworks.

- [ ] Data parallelism / DDP: gradient sync via AllReduce *(P8.1 scaffolded: `scripts/ddp_minimal.py` — TODOs (a) process-group init from torchrun env, (b) the DDP wrap, (c) prove DDP grad == manual allreduce average / grads identical across ranks / replicas identical after each rank's own optimizer.step; helpers `tinyr/trainer/distributed.py` (`shard_for_rank`, `even_len`). Study: `docs/distributed_study.md` — why LoRA makes DDP ~comm-free: 8.7 MB vs 1.0 GB of gradients per step. Acceptance: `torchrun --nproc_per_node=3 scripts/ddp_minimal.py`, three True verdicts per rank — the red marker is the torchrun run, not pytest)*
- [ ] FSDP — why sharding parameters reduces memory pressure *(study `docs/distributed_study.md` §5: the 16 B/param ledger + ZeRO 1/2/3 table — and why TinyRL's LoRA ledger leaves ZeRO nothing to save)*
- [ ] ZeRO stages 1 / 2 / 3 *(§5, same table)*
- [ ] NCCL basics; when communication becomes the bottleneck *(§3: ring AllReduce math `2(N-1)/N`, the 100x LoRA-vs-full-FT comm gap, §4 bucketing/overlap)*
- [ ] Hands-on: a minimal DDP example *(P8.1 = the script above; P8.2 next: engine + DDP trainers on the 3-GPU box, topology decision in study §7)*

> Deferred: tensor parallelism and pipeline parallelism (revisit later).

---

### Phase 9 — Compare TinyRL with verl

Goal: map TinyRL components to a production RL framework.

Trace only the main GRPO path:

```text
dataset
→ rollout worker
→ reward
→ advantage
→ actor update
→ weight synchronization
```

Tasks:

- [ ] Find the actor worker in verl
- [ ] Find the rollout worker
- [ ] Find the reference policy
- [ ] Find reward computation
- [ ] Find GRPO advantage computation
- [ ] Find policy loss
- [ ] Find weight synchronization
- [ ] Find batching logic
- [ ] Find distributed worker orchestration
- [ ] Create `docs/tinyr_vs_verl.md`

---

### Phase 10 — Open Source Contribution

Goal: make at least one real upstream contribution.

Candidate projects:

- verl
- OpenRLHF
- TRL
- vLLM

Possible contributions:

- [ ] Bug fix
- [ ] Unit tests
- [ ] Metrics / profiler improvements
- [ ] Configuration validation
- [ ] Batching edge-case fixes
- [ ] OOM diagnostics
- [ ] Benchmark utilities
- [ ] Documentation fixes with verified code

Milestones:

- [ ] Open first upstream PR
- [ ] Respond to reviewer feedback
- [ ] Get one PR merged

---

## Final Demo

TinyRL is considered resume-ready when it can demonstrate:

- [ ] Complete GRPO training loop
- [ ] Real small-LLM post-training
- [ ] Modular rollout and training components
- [ ] Experience buffering
- [ ] GPU-memory profiling
- [ ] At least one memory optimization
- [ ] Throughput profiling
- [ ] Dynamic batching or padding reduction
- [ ] Rollout/training decoupling
- [ ] Baseline vs. optimized benchmark
- [ ] Architecture comparison with verl
- [ ] At least one upstream RL/AI-Infra PR

Expected final outputs:

- Training loss curve
- Reward curve
- Accuracy curve
- Rollout tokens/s
- Training tokens/s
- Peak GPU memory
- Step-latency breakdown
- Before/after optimization benchmark
- Example model generations

---

## Learning Path

```text
GRPO Algorithm
      ↓
Minimal GRPO Implementation
      ↓
RL Training Data Flow
      ↓
GPU Memory Optimization
      ↓
Profiling
      ↓
Dynamic Batching
      ↓
Rollout / Training Decoupling
      ↓
Inference Engine Concepts
      ↓
DDP / FSDP / ZeRO
      ↓
verl Architecture
      ↓
Open-Source Contribution
```

---

## Scope

TinyRL v0.x intentionally does **not** aim to:

- Implement CUDA kernels from scratch
- Implement NCCL
- Support multi-node training from day one
- Support every RL algorithm
- Train a reward model before the GRPO pipeline works
- Support PPO, GRPO, DPO, and every post-training method simultaneously
- Optimize for 70B-scale models

The initial focus is intentionally narrow:

> **GRPO + single GPU + small LLM + measurable system optimization.**

---

## Status

TinyRL is currently under active development.

The first milestone is to run an end-to-end GRPO training step on `Qwen2.5-0.5B-Instruct` using a single GTX 1080 Ti.

---

## License

TBD

