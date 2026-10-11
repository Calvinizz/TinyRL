"""Phase 7 benchmark: prefill vs decode throughput, per batch size.

Local (1080 Ti): HF backend rows — the prefill/decode asymmetry that
motivates dedicated rollout engines (docs/inference_study.md §1).
Server (390): a vLLM row appears automatically when vllm imports and
VLLMRolloutBackend is implemented (P7.2) — same prompts, same counts.

Regenerate:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      conda run -n tinyrl python scripts/benchmark_rollout.py \
      --batch-sizes 1,2,4,8 --max-new-tokens 128
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyr.config import MODEL_PATH, TrainingConfig
from tinyr.models import ModelManager
from tinyr.rollout import RolloutWorker, VLLMRolloutBackend
from scripts.train_tinyrl import make_toy_prompts

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "rollout.md"


def timeit(fn) -> tuple[float, float]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8")
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sizes = [int(s) for s in args.batch_sizes.split(",")]

    cfg = TrainingConfig(
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    prompts = make_toy_prompts()
    model_mgr = ModelManager(cfg)
    worker = RolloutWorker(model_mgr.policy, model_mgr.tokenizer, cfg)

    rows = []
    for B in sizes:
        torch.cuda.empty_cache()
        questions = [p["question"] for p in prompts[:B]]
        roll = worker.generate(questions)  # warmup + the real decode run

        # -- decode: the autoregressive loop we just timed indirectly;
        # re-run it under the clock for a clean measurement
        decode_s, roll = timeit(lambda: worker.generate(questions))
        resp_tok = roll.attention_mask[:, roll.prompt_width :].sum().item()
        decode_toks = resp_tok / decode_s

        # -- prefill: one no-grad forward over the SAME prompt block the
        # rollout paid serially during its first decode step
        prefill_s, _ = timeit(
            lambda: model_mgr.policy(
                input_ids=roll.prompt_ids,
                attention_mask=roll.attention_mask[:, : roll.prompt_width],
            )
        )
        prompt_tok = roll.attention_mask[:, : roll.prompt_width].sum().item()
        prefill_toks = prompt_tok / prefill_s

        rows.append(
            {
                "batch": f"{B}x{args.group_size}",
                "n": B * args.group_size,
                "prefill": prefill_toks,
                "decode": decode_toks,
                "per_seq": decode_toks / (B * args.group_size),
            }
        )
        print(f"{B}x{args.group_size}: prefill {prefill_toks:.0f} | decode {decode_toks:.0f} tok/s")

    # -- optional vLLM row (server only) --------------------------------
    vllm_row = None
    try:
        import vllm  # noqa: F401

        backend = VLLMRolloutBackend(
            str(MODEL_PATH), model_mgr.tokenizer, cfg
        )
        B = sizes[-1]
        questions = [p["question"] for p in prompts[:B]]
        t, roll = timeit(lambda: backend.generate(questions))
        resp_tok = roll.attention_mask[:, roll.prompt_width :].sum().item()
        vllm_row = {
            "batch": f"{B}x{args.group_size}",
            "n": B * args.group_size,
            "decode": resp_tok / t,
            "per_seq": resp_tok / t / (B * args.group_size),
        }
        print(f"vllm {B}x{args.group_size}: decode {vllm_row['decode']:.0f} tok/s")
    except (ImportError, NotImplementedError) as e:
        print(f"(no vLLM row: {type(e).__name__}: {e})")

    lines = [
        "# Phase 7 — Rollout Backend Benchmark (prefill vs decode)",
        "",
        "Hardware: see row source · Model: Qwen2.5-0.5B-Instruct + LoRA r16 ·",
        "Task: toy arithmetic · `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`",
        "",
        f"Workload: max_new_tokens={args.max_new_tokens}, "
        f"group_size={args.group_size}.",
        "prefill = one no-grad forward over the prompt block (parallel);",
        "decode = the HF/vLLM autoregressive loop (serial). Tokens counted",
        "are REAL (attention-masked), padding excluded.",
        "",
        "## Results",
        "",
        "| backend | batch | rows | prefill tok/s | decode tok/s | per-seq tok/s |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| hf | {r['batch']} | {r['n']} | {r['prefill']:.0f} "
            f"| {r['decode']:.0f} | {r['per_seq']:.1f} |"
        )
    if vllm_row:
        lines.append(
            f"| vllm | {vllm_row['batch']} | {vllm_row['n']} | — "
            f"| {vllm_row['decode']:.0f} | {vllm_row['per_seq']:.1f} |"
        )
    lines += ["", "## Findings", "", "(filled after analysis)", ""]

    BENCH.write_text("\n".join(lines))
    print(f"\nwrote {BENCH}")


if __name__ == "__main__":
    main()
