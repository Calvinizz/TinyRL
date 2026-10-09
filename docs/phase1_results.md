# Phase 1 Results — Minimal GRPO Trainer

Date: 2026-10-09 · Hardware: GTX 1080 Ti 11GB · Env: `tinyrl` (torch 2.7.1+cu118, transformers 5.19, peft 0.21)

Setup: `Qwen2.5-0.5B-Instruct` + LoRA (r=16, α=32, qkvo) with fp32 adapters on a
frozen fp16 base; ref model = frozen base. Group size 2, 2 inner epochs,
clip ε=0.2, KL coef 0.04. Full-FT was not viable on 11GB (fp16 AdamW cannot
represent lr~1e-6 updates; fp32 master weights + Adam do not fit) — see Phase 3.

## Toy task (two-digit addition, 16 prompts)

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train_grpo.py --toy --max-steps 30 --lr 1e-4
```

| metric | value |
|---|---|
| reward, first 10 steps | 0.925 |
| reward, last 10 steps | 0.950 (acc saturated at 1.00 from step 2) |
| final KL(policy ‖ ref) | 0.044 |
| step latency | 2–9 s |

Checklist: **learning verified** — accuracy saturated, gradients non-zero
(96/192 LoRA params on step 1; B-matrices start at zero), clipping active
early (clip_frac up to 0.108) then settling to ~0 as ratio returned to band.

## GSM8K subset (500 prompts)

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train_grpo.py --max-steps 50 --n-prompts-per-step 3
```

(batch reduced 4→3 prompts: 4×2 OOMs at ~T=450 — the fp32
`[N, T, 151936]` log_softmax saved for backward is the memory bottleneck)

| metric | value |
|---|---|
| reward, first 10 steps | 0.283 |
| reward, last 10 steps | 0.367 |
| mean reward | 0.333 |
| final KL | 0.008 |
| step latency | ~21 s |

Observations:

- Reward is noisy per step (batch = 6 responses); a 50-step run is a
  smoke test, not a convergence claim.
- Steps with uniform group rewards show loss ≈ 0 and grad ≈ 0 — correct
  GRPO behavior (no within-group signal ⇒ no gradient).
- KL grows slowly; ratio stays in [0.84, 1.15] — updates conservative at
  lr 2e-5, clipping rarely engages.

## Reward-function lessons (debugging notes)

1. Model often ignores the `#### N` format at temperature 1.0 → reward
   silently all-zero → advantages all-zero → zero gradients, no error.
   Fixed with an example in the system prompt + a last-number fallback in
   `extract_answer`.
2. `old`/`ref` log-prob passes must run under `torch.no_grad()` — a stray
   autograd graph keeps the fp32 log_softmax alive until backward and OOMs
   an 11GB GPU.

Raw logs: `logs/toy_grpo.log`, `logs/gsm8k_grpo.log` (not committed).
