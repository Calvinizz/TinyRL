"""Phase 3 — memory benchmark harness.

Runs the toy task under each Phase-3 optimization flag and records peak
CUDA memory + wall time per step into benchmarks/memory.md.

Run (after TODOs P3.1–P3.4 are implemented):

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        conda run -n tinyrl python scripts/benchmark_memory.py --max-steps 5

Memory-measurement gotcha this script must respect:
torch.cuda.max_memory_allocated() is a process-lifetime high-water mark —
it never comes down on its own. All variants run in one process, so each
variant MUST reset the peak stats first or it reports the max over every
earlier variant (they all come out identical).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tinyr.config import TrainingConfig  # noqa: E402
from train_tinyrl import make_toy_prompts, train  # noqa: E402

# (name, config overrides) — the ladder of Phase-3 optimizations
VARIANTS = [
    ("baseline", {}),
    ("gradient_checkpointing", {"gradient_checkpointing": True}),
    ("ref_cpu_offload", {"ref_model_offload": True}),
    ("micro_batch_2", {"micro_batch_size": 2}),
    (
        "everything",
        {
            "gradient_checkpointing": True,
            "ref_model_offload": True,
            "micro_batch_size": 2,
        },
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-prompts-per-step", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()

    prompts = make_toy_prompts()
    rows = [
        "| variant | peak alloc (GiB) | s/step | notes |",
        "|---|---|---|---|",
    ]

    for name, overrides in VARIANTS:
        cfg = TrainingConfig(
            max_steps=args.max_steps,
            lr=args.lr,
            n_prompts_per_step=args.n_prompts_per_step,
            max_new_tokens=args.max_new_tokens,
            **overrides,
        )
        # return the previous variant's model memory to the driver and
        # restart the high-water mark — otherwise this variant would
        # report the peak of everything that ran before it
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        train(cfg, prompts)
        elapsed = time.perf_counter() - t0
        peak_gib = torch.cuda.max_memory_allocated() / 1024 ** 3
        sps = elapsed / args.max_steps
        print(f"[benchmark] {name}: peak {peak_gib:.3f} GiB, {sps:.1f} s/step")
        rows.append(f"| {name} | {peak_gib:.3f} | {sps:.1f} | |")

    out = ROOT / "benchmarks" / "memory.md"
    out.parent.mkdir(exist_ok=True)
    header = (
        "# Phase 3 — Memory Benchmark\n\n"
        "Hardware: GTX 1080 Ti 11GB · Model: Qwen2.5-0.5B-Instruct + LoRA r16\n"
        "(fp32 adapters on frozen fp16 base) · Task: toy arithmetic ·\n"
        "`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`\n\n"
        f"Workload: {args.n_prompts_per_step} prompts × group 2, "
        f"max_new_tokens={args.max_new_tokens}, {args.max_steps} steps.\n\n"
        "Regenerate with:\n\n"
        "```bash\n"
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\\n"
        "  conda run -n tinyrl python scripts/benchmark_memory.py "
        "--max-steps 5 \\\n"
        f"  --n-prompts-per-step {args.n_prompts_per_step} "
        f"--max-new-tokens {args.max_new_tokens}\n"
        "```\n\n"
    )
    # regenerate the whole file every run — a hand-written Findings section
    # below the table is preserved by re-appending it after regeneration
    findings = ""
    if out.exists():
        text = out.read_text()
        if "## Findings" in text:
            findings = "\n" + text.split("## Findings", 1)[1].lstrip("\n")
            findings = "## Findings" + findings
    out.write_text(header + "## Results\n\n" + "\n".join(rows) + "\n" + findings)
    print(f"[benchmark] wrote {out}")


if __name__ == "__main__":
    main()
