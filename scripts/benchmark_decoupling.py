"""Phase 6 benchmark: sync lockstep vs async producer/consumer.

Runs scripts/train_async.py's train() in both modes with identical
config/seeds and writes benchmarks/decoupling.md. The interesting
columns are wall time (did overlap pay?) and lag/drops (what did
off-policy-ness cost?).

Regenerate:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      conda run -n tinyrl python scripts/benchmark_decoupling.py \
      --max-steps 8 --n-prompts-per-step 4 --micro-batch-size 4
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyr.config import TrainingConfig
from scripts.train_async import train
from scripts.train_tinyrl import make_toy_prompts

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "decoupling.md"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--n-prompts-per-step", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--max-staleness", type=int, default=3)
    args = parser.parse_args()

    prompts = make_toy_prompts()
    rows = []
    for mode in ("sync", "async"):
        cfg = TrainingConfig(
            n_prompts_per_step=args.n_prompts_per_step,
            group_size=args.group_size,
            micro_batch_size=args.micro_batch_size,
            max_new_tokens=args.max_new_tokens,
            max_steps=args.max_steps,
            queue_size=args.queue_size,
            max_staleness=args.max_staleness,
            seed=0,
        )
        # fresh CUDA state per mode — peak memory must be per-run
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        out = train(cfg, prompts, mode)
        lags = out["lags"]
        rows.append(
            {
                "mode": mode,
                "wall": out["wall"],
                "tok_s": out["resp_tokens"] / out["wall"],
                "steps": out["steps"],
                "drops": out["drops"],
                "lag_mean": sum(lags) / len(lags) if lags else 0.0,
                "lag_max": max(lags, default=0),
                "peak": out["peak_gib"],
            }
        )

    lines = [
        "# Phase 6 — Sync vs Async Decoupling Benchmark",
        "",
        "Hardware: GTX 1080 Ti 11GB · Model: Qwen2.5-0.5B-Instruct + LoRA r16 ·",
        "Task: toy arithmetic · `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`",
        "",
        f"Workload: {args.max_steps} steps, batch "
        f"{args.n_prompts_per_step}x{args.group_size}, max_new_tokens="
        f"{args.max_new_tokens}, micro_batch_size={args.micro_batch_size},",
        f"queue_size={args.queue_size}, max_staleness={args.max_staleness}.",
        "Both modes share one code path (queue -> stamp -> gate -> train),",
        "same seed, model reloaded per run.",
        "",
        "Regenerate:",
        "",
        "```bash",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\",
        "  conda run -n tinyrl python scripts/benchmark_decoupling.py \\",
        f"  --max-steps {args.max_steps} --n-prompts-per-step "
        f"{args.n_prompts_per_step} --micro-batch-size {args.micro_batch_size}",
        "```",
        "",
        "## Results",
        "",
        "| mode | wall (s) | end-to-end tok/s | steps trained | drops "
        "| lag mean | lag max | peak (GiB) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['mode']} | {r['wall']:.1f} | {r['tok_s']:.0f} "
            f"| {r['steps']} | {r['drops']} | {r['lag_mean']:.2f} "
            f"| {r['lag_max']} | {r['peak']:.2f} |"
        )
    lines += ["", "## Findings", "", "(filled after analysis)", ""]

    BENCH.write_text("\n".join(lines))
    print(f"\nwrote {BENCH}")


if __name__ == "__main__":
    main()
