# Phase 8 — DDP trainers under a live engine

scripts/train_ddp.py: sidecar topology (study §7 option B) — vLLM
engine alone on GPU 0, a torchrun world of 2 DDP ranks on GPUs 1,2,
joined by a remote-manager queue pair. Rank 0 reads experiences,
`broadcast_object_list` fans them out (numpy payloads — see Finding 4),
all ranks run the identical staleness verdict, shard at the PROMPT
level (`group_rows_for_rank`), train half the batch each, and rank 0
ships its post-AllReduce LoRA to the engine per step (P7.3 machinery,
unchanged). Protocol matches benchmarks/rematch.md: toy task, batch
4x2, queue 2, max_staleness 3, 10 steps, RTX 3090, 2026-10-12.

## Results

| leg | trainers | wall/step (raw) | wall/step (steady) | resp tok/s (steady) | lag mean/max | drops | trainer peak |
|---|---|---|---|---|---|---|---|
| rematch vllm-async-2gpu | 1 (8 rows) | 0.71 s | 0.71 s | 381.3 | 2.4 / 3 | 0 | 9.87 GiB |
| P8.2 sidecar DDP | 2 ranks (4 rows each) | 4.66 s | **0.70 s** | **389.8** | 2.4 / 3 | 0 | 5.88 GiB / rank |

The rematch leg builds its engine BEFORE `wall_start`
(train_async.py:94 vs :208), so its clock never sees the boot; the
DDP parent starts trainer wall before the first `exp_q.get()`, which
blocks 39.60 s waiting for the engine to come up. "Steady" = (wall −
first-get wait)/steps — the rematch-comparable number; "raw" is what
a 10-step run actually costs end to end. In any real run the ~40 s
boot amortizes to noise.

Per-phase steady-state means (rank 0): bcast 0.02 s, prep 0.12 s,
train 0.42 s, sync 0.05 s — plus ~0.1 s loop overhead ≈ 0.70 s.

## Findings

**Finding 1 — parity, and that is the point.** Steady state 0.70
s/step vs the single-GPU trainer's 0.71: DDP-2 neither wins nor loses
wall/step at toy scale. Each rank's compute halved (4 rows vs 8: prep
0.12 s + train 0.42 s), and the LoRA gradient AllReduce is invisible
(8.7 MB over PCIe, the P8.1 mirror of P7's weight-sync finding) — but
the per-step serial chain grew by ~0.15 s (object broadcast, the
scalar token-count allreduce, the timing barrier, rank-0's save). Net
wash. The win is structural: trainer-side throughput now SCALES with
GPUs, and rematch Finding 4 already showed the trainer is the next
bottleneck (rollout got cheaper than prepare+train in Phase 7). The
first leg where single-GPU prep+train exceeds ~0.6 s/step is where
this table flips.

**Finding 2 — the invariant chain held in production.** 0 drops with
lag pinned at mean 2.4 / max 3: queue 2 + staleness 3 is the ceiling,
and the world never desynced — every rank computed the identical
staleness verdict from the identical broadcast experience (the
comment in train_ddp.py's gate is the hang-prevention contract).
Per-rank peaks identical to 0.01 GiB (5.88/5.88): same shard size,
same bucket schedule. P8.1's receipts (grad rel_err 6.35e-08,
bitwise-identical weights across ranks) are why nobody had to think
about replica drift during this run.

**Finding 3 — zero gradients at step 1 were correct.** The log opens
with `grads non-zero: 0/192` on both ranks: the toy task's first
three batches were all-correct (reward 1.000 ×8) → within-group
std = 0 → advantage ≡ 0; and with a fresh LoRA (lora_B = 0) the
policy IS the reference → kl ≡ 0. Loss identically zero, gradients
exactly zero — GRPO has no signal when a group is unanimous.
Learning starts at step 4 (reward 0.875) and the kl/clip columns
wake up (0.0011 / 0.013 — clip ≠ 0 requires n_inner_epochs ≥ 2, the
second inner epoch's ratios moving off 1). verify_gradients earns
its keep: the distinction between "degenerate batch" and "broken
mask" is exactly what it exists to flag.

**Finding 4 — torch tensors do not cross process trees.** Runs 1-2
died on the IPC boundary. torch installs custom pickle reducers on
tensors: pickling one for another process defaults to shared-memory
fd passing (multiprocessing.reductions → rebuild_storage_fd), and
that channel's authkey challenge uses each process's OWN key — a
remote-manager server and an unrelated engine subprocess can never
shake hands ("AuthenticationError: digest sent was rejected"; run 1
also showed mp.Queue behind a manager hitting the same resource_sharer
path). The fix is the general law: **the boundary carries bytes, not
tensors** — `exp_to_payload` strips to numpy, `payload_to_experience`
rebuilds on each rank's own GPU (also fixing a latent bug: it
replaced an `_to_device` that assigned the read-only `input_ids`
property). Same law as P7's weight sync shipping safetensors files.

**Finding 5 — startup hermeticity is part of the interface.** Run 3
died inside flashinfer's JIT (nvcc 10.1 cannot parse
`-generate-dependencies-with-compile`) because the parent launched
children from a bare environment: the two VLLM_* pins that
tests/test_backend.py already self-applies now ship in the parent's
env dict too (`setdefault` — shell values still win). A script that
only works after `source server_env.sh` is a script with an invisible
dependency; the runbook in its own docstring is not an API.
