"""Phase 5 — padding benchmark: fixed vs length-bucketed micro-batching.

Runs the toy task twice per batch width (dynamic_batching off/on) and
writes benchmarks/padding.md: padding ratio, train section time, train
tok/s, peak memory.

Run:

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        conda run -n tinyrl python scripts/benchmark_padding.py \\
        --max-steps 5 --n-prompts 4,8 --micro-batch-size 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tinyr.config import TrainingConfig  # noqa: E402
from train_tinyrl import make_toy_prompts, train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-prompts", type=str, default="4,8")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    args = parser.parse_args()
    sweep = [int(v) for v in args.n_prompts.split(",")]

    prompts = make_toy_prompts()
    rows = [
        "| batch | variant | pad (batch) | train s/step | train tok/s "
        "| peak (GiB) |",
        "|---|---|---|---|---|---|",
    ]

    for n in sweep:
        for name, dynamic in [("fixed", False), ("bucketed", True)]:
            cfg = TrainingConfig(
                max_steps=args.max_steps,
                lr=args.lr,
                n_prompts_per_step=n,
                max_new_tokens=args.max_new_tokens,
                micro_batch_size=args.micro_batch_size,
                dynamic_batching=dynamic,
            )
            stats = train(cfg, prompts)
            summary = stats["timer"].summary()
            steps = stats["steps"]
            trn_tps = stats["resp_tokens"] / (summary["train"] * steps)
            rows.append(
                f"| {n}×{cfg.group_size} | {name} "
                f"| {stats['pad_ratio']:.0%} "
                f"| {summary['train']:.2f} | {trn_tps:.0f} "
                f"| {stats['peak_gib']:.2f} |"
            )
            print(
                f"[padding] batch {n}x{cfg.group_size} {name}: "
                f"pad {stats['pad_ratio']:.0%}, "
                f"train {trn_tps:.0f} tok/s, peak {stats['peak_gib']:.2f} GiB"
            )

    out = ROOT / "benchmarks" / "padding.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        "# Phase 5 — Padding Benchmark\n\n"
        "Hardware: GTX 1080 Ti 11GB · Model: Qwen2.5-0.5B-Instruct + LoRA r16\n"
        "(fp32 adapters on frozen fp16 base) · Task: toy arithmetic ·\n"
        "`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`\n\n"
        f"Workload: {args.max_steps} steps per config, "
        f"max_new_tokens={args.max_new_tokens}, "
        f"micro_batch_size={args.micro_batch_size}.\n"
        "`pad (batch)` is the response-block padding ratio of the FULL\n"
        "batch (identical for both variants — bucketing changes what each\n"
        "MICRO-BATCH pays, not the batch itself).\n\n"
        "Regenerate with:\n\n"
        "```bash\n"
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\\n"
        "  conda run -n tinyrl python scripts/benchmark_padding.py \\\n"
        f"  --max-steps {args.max_steps} --n-prompts {args.n_prompts} \\\n"
        f"  --micro-batch-size {args.micro_batch_size}\n"
        "```\n\n"
        "## Results\n\n" + "\n".join(rows) + "\n\n## Findings\n\n"
        "(fill in after the run)\n"
    )
    print(f"[padding] wrote {out}")


if __name__ == "__main__":
    main()
