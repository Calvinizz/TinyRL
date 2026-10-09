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

- [ ] Implement `RolloutWorker`
- [ ] Implement `RewardManager`
- [ ] Implement `Experience`
- [ ] Implement `ExperienceBuffer`
- [ ] Implement `GRPOTrainer`
- [ ] Implement `ModelManager`
- [ ] Separate rollout logic from training logic

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

- [ ] Add shape assertions
- [ ] Add dtype assertions
- [ ] Add device assertions
- [ ] Add unit tests for core components

---

### Phase 3 — Memory Optimization

Goal: make GRPO practical on an 11GB GPU.

#### Learn

- [ ] Parameter memory
- [ ] Gradient memory
- [ ] Optimizer-state memory
- [ ] Activation memory
- [ ] KV-cache memory
- [ ] FP32 vs. FP16 memory usage
- [ ] Gradient accumulation
- [ ] Activation checkpointing
- [ ] CPU offload

#### Implement

- [ ] FP16 training
- [ ] LoRA / PEFT support
- [ ] Gradient accumulation
- [ ] Gradient checkpointing
- [ ] Configurable micro-batch size
- [ ] Reference-model CPU offload
- [ ] Peak GPU-memory profiling
- [ ] OOM diagnostics
- [ ] Group-size fallback for low-memory hardware

#### Benchmark

- [ ] Record baseline peak memory
- [ ] Record optimized peak memory
- [ ] Compare full fine-tuning vs. LoRA if feasible
- [ ] Document results in `benchmarks/memory.md`

---

### Phase 4 — Throughput Profiling

Goal: identify where RL post-training spends time.

#### Learn

- [ ] Rollout latency
- [ ] Forward latency
- [ ] Backward latency
- [ ] Optimizer-step latency
- [ ] Tokens per second
- [ ] GPU utilization
- [ ] Padding waste

#### Implement

- [ ] Measure rollout time
- [ ] Measure reward time
- [ ] Measure log-probability recomputation time
- [ ] Measure backward time
- [ ] Measure optimizer-step time
- [ ] Measure rollout tokens/s
- [ ] Measure training tokens/s
- [ ] Measure total step latency
- [ ] Log peak GPU memory

Target metrics:

```text
rollout_time
reward_time
logprob_time
backward_time
optimizer_time
rollout_tokens_per_second
train_tokens_per_second
peak_gpu_memory
```

- [ ] Document results in `benchmarks/throughput.md`

---

### Phase 5 — Dynamic Batching & Padding Reduction

Goal: reduce wasted computation caused by variable sequence lengths.

#### Learn

- [ ] Padding overhead
- [ ] Sequence-length imbalance
- [ ] Token-based batching
- [ ] Dynamic batching

#### Implement

- [ ] Log response-length distribution
- [ ] Bucket samples by response length
- [ ] Implement dynamic micro-batching
- [ ] Measure padding ratio
- [ ] Compare fixed-size and dynamic batching
- [ ] Measure throughput improvement

---

### Phase 6 — Rollout / Training Decoupling

Goal: understand a core RL-system design pattern.

#### Learn

- [ ] Synchronous RL training
- [ ] Rollout-training decoupling
- [ ] Producer-consumer architecture
- [ ] Policy staleness
- [ ] On-policy vs. off-policy behavior in system design

#### Implement

- [ ] Separate rollout and training modules
- [ ] Introduce an experience queue
- [ ] Make rollout produce experiences
- [ ] Make trainer consume experiences
- [ ] Track policy version per experience
- [ ] Log policy-version lag
- [ ] Add a maximum staleness threshold
- [ ] Build a synchronous baseline
- [ ] Build a simple asynchronous prototype
- [ ] Compare throughput and utilization

---

### Phase 7 — Inference Backend Abstraction

Goal: make rollout generation independent from the trainer implementation.

#### Learn

- [ ] Hugging Face `generate()`
- [ ] Prefill vs. decode
- [ ] KV cache
- [ ] Continuous batching
- [ ] PagedAttention
- [ ] Why RL systems use dedicated rollout engines

#### Implement

- [ ] Define a common rollout backend interface
- [ ] Implement a Hugging Face backend
- [ ] Keep the trainer independent from the rollout backend
- [ ] Investigate vLLM integration
- [ ] Document GTX 1080 Ti compatibility limitations

Example interface:

```python
class RolloutBackend:
    def generate(self, prompts, **kwargs):
        raise NotImplementedError
```

---

### Phase 8 — Distributed Training Fundamentals

Goal: understand the concepts used by production-grade RL training frameworks.

#### Learn

- [ ] Data Parallelism
- [ ] DDP
- [ ] AllReduce
- [ ] ReduceScatter
- [ ] AllGather
- [ ] FSDP
- [ ] ZeRO Stage 1
- [ ] ZeRO Stage 2
- [ ] ZeRO Stage 3
- [ ] Tensor Parallelism
- [ ] Pipeline Parallelism
- [ ] NCCL basics
- [ ] PCIe vs. NVLink

#### Practice

- [ ] Write a minimal DDP example
- [ ] Understand gradient synchronization
- [ ] Read a minimal FSDP example
- [ ] Explain why FSDP reduces memory pressure
- [ ] Explain when communication becomes the bottleneck

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

