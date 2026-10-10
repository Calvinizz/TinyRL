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

- [ ] Step-latency breakdown: rollout / logprob forward+backward / optimizer step
- [ ] Rollout tokens/s and training tokens/s
- [ ] Peak GPU memory logged per step
- [ ] Record results in `benchmarks/throughput.md`

---

### Phase 5 — Dynamic Batching & Padding Reduction

Goal: stop paying compute for padding tokens.

- [ ] Measure padding ratio and the response-length distribution
- [ ] Bucket samples by response length
- [ ] Dynamic micro-batching driven by the buckets
- [ ] Compare fixed vs dynamic batching throughput

---

### Phase 6 — Rollout / Training Decoupling

Goal: the core RL-system design pattern — producer/consumer with
staleness control.

- [ ] Experience queue between rollout and trainer
- [ ] Rollout worker produces experiences; trainer consumes them
- [ ] Policy version stamped on every experience
- [ ] Staleness: log version lag, cap it with a max-staleness threshold
- [ ] Sync baseline vs simple async prototype: throughput comparison

---

### Phase 7 — Inference Backend Abstraction

Goal: the trainer shouldn't know how generation happens.

- [ ] `RolloutBackend` interface (`generate(prompts, **kwargs)`)
- [ ] Hugging Face `generate()` backend behind it; trainer stays backend-agnostic
- [ ] Study: prefill vs decode, KV cache, continuous batching, PagedAttention — why RL systems use dedicated rollout engines

```python
class RolloutBackend:
    def generate(self, prompts, **kwargs):
        raise NotImplementedError
```

> Deferred: real vLLM integration — vLLM requires compute capability
> >= 7.0 (Volta), so the GTX 1080 Ti (Pascal, sm_61) cannot run it;
> document the limitation instead.

---

### Phase 8 — Distributed Training Fundamentals

Goal: the vocabulary needed to read production RL frameworks.

- [ ] Data parallelism / DDP: gradient sync via AllReduce
- [ ] FSDP — why sharding parameters reduces memory pressure
- [ ] ZeRO stages 1 / 2 / 3
- [ ] NCCL basics; when communication becomes the bottleneck
- [ ] Hands-on: a minimal DDP example

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

