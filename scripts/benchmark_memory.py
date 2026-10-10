"""Phase 3 — memory benchmark harness.

Runs the toy task under each Phase-3 optimization flag and records peak
CUDA memory + wall time per step into benchmarks/memory.md.

Run (after TODOs P3.1–P3.4 are implemented):

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        conda run -n tinyrl python scripts/benchmark_memory.py --max-steps 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tinyr.config import TrainingConfig  # noqa: E402
from tinyr.profiling import MemoryProfiler  # noqa: E402
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
    args = parser.parse_args()

    prompts = make_toy_prompts()
    rows = ["| variant | peak alloc (GiB) | s/step | notes |",
            "|---|---|---|---|"]

    for name, overrides in VARIANTS:
        cfg = TrainingConfig(
            max_steps=args.max_steps, lr=args.lr, **overrides
        )
        prof = MemoryProfiler()
        prof.mark("after setup")
        t0 = time.perf_counter()
        train(cfg, prompts)
        elapsed = time.perf_counter() - t0
        prof.mark("after training")
        peak_gib = prof.peak / 1024 ** 3
        sps = elapsed / args.max_steps
        print(f"[benchmark] {name}: peak {peak_gib:.3f} GiB, {sps:.1f} s/step")
        rows.append(f"| {name} | {peak_gib:.3f} | {sps:.1f} | |")

    out = ROOT / "benchmarks" / "memory.md"
    out.parent.mkdir(exist_ok=True)
    header = out.read_text().split("## Results")[0] if out.exists() else (
        "# Phase 3 — Memory Benchmark\n\n"
        "GTX 1080 Ti · Qwen2.5-0.5B-Instruct + LoRA r16 (fp32 adapters) · "
        "toy task · 4×2 batch · expandable_segments:True\n\n"
    )
    out.write_text(header + "## Results\n\n" + "\n".join(rows) + "\n")
    print(f"[benchmark] wrote {out}")


if __name__ == "__main__":
    main()
