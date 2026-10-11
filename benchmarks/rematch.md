# Phase 6 Rematch — decoupling on dedicated hardware

The Phase-6 verdict (benchmarks/decoupling.md, GTX 1080 Ti) was
"single-GPU async is a WASH": 46 tok/s in both modes. This is the
appeal on the 390 server — same protocol (toy task, batch 4x2,
max_new_tokens 256, queue 2, max_staleness 3, 10 steps), one variable
per leg, RTX 3090, 2026-10-11.

## Results

| leg | rollout | GPU layout | wall/step | resp tok/s | lag mean/max | drops | trainer peak |
|---|---|---|---|---|---|---|---|
| sync | HF generate() | 1 | 8.93 s | 39.2 | 0 / 0 | 0 | 9.87 GiB |
| async | HF generate() | 1 | 5.63 s | 49.6 | 0.9 / 1 | 0 | 9.00 GiB |
| async | vLLM engine | shared GPU 0 | 0.74 s | 351.4 | 2.1 / 3 | 0 | 9.87 GiB |
| async | vLLM engine | engine GPU 0 + trainer GPU 1 | **0.71 s** | **381.3** | 2.4 / 3 | 0 | 9.87 GiB (cuda:1) |

(vLLM legs include live weight sync — every optimizer step saves the
LoRA and hot-swaps it into the engine via a versioned LoRARequest.)

## Findings

**Finding 1 — the wash did not survive the hardware change.** HF
async on the 3090 is +26% tok-s (+37% wall) over sync, where the
1080 Ti measured a tie. The P6 tie was a property of that box's
BALANCE — decode and training both host-bound on a weak CPU, nothing
to overlap into. On the 3090 the trainer's kernels are fast and the
rollout is still host-bound, so training fills the decode loop's
CPU gaps even on one shared stream. Lesson: "single-GPU async is a
wash" is not a law; it is a symptom of both sides fighting for the
same host.

**Finding 2 — the engine dwarfs the topology question.** The vLLM
legs beat HF lockstep by 12.6x wall / 9.7x tok-s. The decisive
change was never "which GPU runs what" but "who runs the decode
loop" (Phase 7's whole thesis, reconfirmed under training load).

**Finding 3 — the dedicated second GPU buys ~nothing at toy scale**
(+4% tok-s, wall 0.74 -> 0.71 s, within noise). Engine and trainer
are separate PROCESSES — each with its own CUDA context — so sharing
one 3090 by time-slicing already overlaps them when neither
saturates silicon. A dedicated card pays when one side saturates
(bigger model, bigger batches, TP), or when memory isolation
matters; at 4x2 x 0.5B both cards sit ~40% loaded. The cross-card
cost that DOES exist — experiences copied GPU 0 -> GPU 1 over PCIe
(~2600 tokens/step, microseconds) — is invisible at this scale.

**Finding 4 — the bottleneck flipped to the trainer.** In the vLLM
legs the producer runs ~2.1-2.4 steps AHEAD (lag mean; the queue at
2 plus max_staleness 3 is the only thing stopping it running
further ahead, and it never dropped a batch). Rollout per step is
now CHEAPER than prepare+train — the exact inversion of the Phase
2-6 economics, where rollout was 80%+ of step time. That inversion
is Phase 8's reason to exist: the next marginal tok/s comes from
DDP-ing the trainer, not from speeding the engine further.

**Finding 5 — weight sync is (so far) free.** ~9 MiB adapter save +
engine hot-swap per step, inside the 0.71 s/step wall. At realistic
sync frequencies this cost grows sublinearly (the adapter is tiny);
the expensive strategies (full state_dict push, merge-and-reload)
remain what docs/inference_study.md §7 says they are: for when you
cannot express your update as an adapter.
