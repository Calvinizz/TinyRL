"""Phase 4 — throughput benchmark: where does the step time go?

Runs the toy task at several batch widths, collects the Timer's
per-section means, and writes benchmarks/throughput.md.

Run:

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        conda run -n tinyrl python scripts/benchmark_throughput.py \\
        --max-steps 5 --n-prompts 4,8
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

SECTIONS = ["rollout", "reward", "prepare", "train"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--n-prompts",
        type=str,
        default="4,8",
        help="comma-separated n_prompts_per_step values to sweep",
    )
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()
    sweep = [int(v) for v in args.n_prompts.split(",")]

    prompts = make_toy_prompts()
    sec_rows = [
        "| batch | rollout | reward | prepare | train | Σ measured |",
        "|---|---|---|---|---|---|",
    ]
    tps_rows = [
        "| batch | gen tok/s | train tok/s | gen : train |",
        "|---|---|---|---|",
    ]

    for n in sweep:
        cfg = TrainingConfig(
            max_steps=args.max_steps,
            lr=args.lr,
            n_prompts_per_step=n,
            max_new_tokens=args.max_new_tokens,
        )
        stats = train(cfg, prompts)
        summary = stats["timer"].summary()
        steps = stats["steps"]
        means = {k: summary.get(k, 0.0) for k in SECTIONS}
        total = sum(means.values())

        # tok/s over the WHOLE run: total response tokens / total section
        # time (mean s/step x steps) — averages out per-step length noise
        gen_tps = stats["resp_tokens"] / (means["rollout"] * steps)
        trn_tps = stats["resp_tokens"] / (means["train"] * steps)

        sec_rows.append(
            f"| {n}×{cfg.group_size} | " +
            " | ".join(f"{means[k]:.2f}" for k in SECTIONS) +
            f" | {total:.2f} |"
        )
        tps_rows.append(
            f"| {n}×{cfg.group_size} | {gen_tps:.0f} | {trn_tps:.0f} "
            f"| {trn_tps / gen_tps:.0f}x |"
        )
        print(
            f"[throughput] batch {n}x{cfg.group_size}: "
            f"gen {gen_tps:.0f} tok/s, train {trn_tps:.0f} tok/s"
        )

    out = ROOT / "benchmarks" / "throughput.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        "# Phase 4 — Throughput Benchmark\n\n"
        "Hardware: GTX 1080 Ti 11GB · Model: Qwen2.5-0.5B-Instruct + LoRA r16\n"
        "(fp32 adapters on frozen fp16 base) · Task: toy arithmetic ·\n"
        "`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`\n\n"
        f"Workload: {args.max_steps} steps per config, "
        f"max_new_tokens={args.max_new_tokens}, sweep over batch width.\n\n"
        "Regenerate with:\n\n"
        "```bash\n"
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\\n"
        "  conda run -n tinyrl python scripts/benchmark_throughput.py \\\n"
        f"  --max-steps {args.max_steps} --n-prompts {args.n_prompts}\n"
        "```\n\n"
        "## Results — section time (mean s/step)\n\n"
        + "\n".join(sec_rows)
        + "\n\n## Results — throughput\n\n"
        + "\n".join(tps_rows)
        + "\n\n## Findings\n\n"
        "(fill in after the run — questions in the README order)\n"
    )
    print(f"[throughput] wrote {out}")


if __name__ == "__main__":
    main()
