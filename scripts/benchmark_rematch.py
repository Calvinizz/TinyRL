"""Phase-6 rematch: does decoupling pay once rollout gets its own GPU?

The Phase-6 verdict (benchmarks/decoupling.md, 1080 Ti) was "single-GPU
async is a WASH": producer and consumer threads serialize on the
default CUDA stream, so overlapping them buys nothing. The rematch
changes one variable per leg, all on the same 390 box (RTX 3090):

  hf-sync-1gpu     lockstep, HF generate() — the baseline loop
  hf-async-1gpu    threads, one GPU — the P6 wash, replicated on 3090
  vllm-async-1gpu  engine + trainer SHARE GPU 0 (separate processes,
                   time-sliced — overlap without isolation)
  vllm-async-2gpu  engine GPU 0, trainer cuda:1 — the true rematch:
                   nothing shares silicon, the queue is the only coupling

Protocol = Phase 6's: toy task, batch 4x2, queue 2, max_staleness 3,
10 steps. One leg per process invocation (the vLLM engine must not
leak across legs; each boot ~45 s warm).

Run each leg from the repo root on the server:
    CUDA_VISIBLE_DEVICES=0    python scripts/benchmark_rematch.py hf-sync-1gpu
    CUDA_VISIBLE_DEVICES=0    python scripts/benchmark_rematch.py hf-async-1gpu
    CUDA_VISIBLE_DEVICES=0    python scripts/benchmark_rematch.py vllm-async-1gpu
    CUDA_VISIBLE_DEVICES=0,1  python scripts/benchmark_rematch.py vllm-async-2gpu

The last line on stdout is `REMATCH_JSON {...}` — the assembler's
contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_async import train  # noqa: E402
from scripts.train_tinyrl import make_toy_prompts  # noqa: E402
from tinyr.config import TrainingConfig  # noqa: E402

# leg -> (mode, rollout_backend, trainer device); the engine always
# takes the first VISIBLE device, CUDA_VISIBLE_DEVICES decides which
# physical card that is
LEGS = {
    "hf-sync-1gpu": ("sync", "hf", "cuda:0"),
    "hf-async-1gpu": ("async", "hf", "cuda:0"),
    "vllm-async-1gpu": ("async", "vllm", "cuda:0"),
    "vllm-async-2gpu": ("async", "vllm", "cuda:1"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("leg", choices=LEGS)
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()

    mode, backend, device = LEGS[args.leg]
    cfg = TrainingConfig(
        rollout_backend=backend,
        device=device,
        max_steps=args.max_steps,
    )
    stats = train(cfg, make_toy_prompts(), mode)

    lags = stats["lags"]
    print(
        "REMATCH_JSON "
        + json.dumps(
            {
                "leg": args.leg,
                "steps": stats["steps"],
                "drops": stats["drops"],
                "lag_mean": round(sum(lags) / len(lags), 2) if lags else 0.0,
                "lag_max": max(lags, default=0),
                "wall_s": round(stats["wall"], 2),
                "resp_tokens": stats["resp_tokens"],
                "resp_tok_s": round(stats["resp_tokens"] / stats["wall"], 1),
                "peak_gib": round(stats["peak_gib"], 2),
            }
        )
    )


if __name__ == "__main__":
    main()
