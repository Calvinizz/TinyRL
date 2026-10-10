"""Phase 2 — modular GRPO training entry point.

Wires the tinyr package together; the same dataflow as Phase 1 but with
rollout / reward / experience / training separated:

    prompts -> RolloutWorker -> RewardManager -> Experience
            -> ExperienceBuffer -> GRPOTrainer.prepare -> train_step

Run:
    python scripts/train_tinyrl.py --toy --max-steps 30 --lr 1e-4
    python scripts/train_tinyrl.py --max-steps 50 --n-prompts-per-step 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyr.config import DATA_PATH, TrainingConfig
from tinyr.experience import Experience, ExperienceBuffer
from tinyr.models import ModelManager
from tinyr.profiling import Timer
from tinyr.reward import RewardManager
from tinyr.rollout import RolloutWorker
from tinyr.trainer import GRPOTrainer


def load_prompts(limit: int | None = None) -> list[dict]:
    if not DATA_PATH.exists():
        raise SystemExit(
            f"{DATA_PATH} not found — run `python scripts/prepare_gsm8k.py` first"
        )
    records = [json.loads(line) for line in DATA_PATH.open()]
    return records[:limit] if limit else records


def make_toy_prompts(n: int = 16) -> list[dict]:
    rng = random.Random(1234)
    prompts = []
    for _ in range(n):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        prompts.append({"question": f"What is {a} + {b}?", "answer": str(a + b)})
    return prompts


def train(cfg: TrainingConfig, prompts: list[dict]) -> dict:
    """Run the loop; returns {"timer", "resp_tokens", "steps"} for benchmarks."""
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    model_mgr = ModelManager(cfg)
    rollout_worker = RolloutWorker(model_mgr.policy, model_mgr.tokenizer, cfg)
    reward_mgr = RewardManager()
    trainer = GRPOTrainer(model_mgr.policy, model_mgr.ref, cfg)
    buffer = ExperienceBuffer(max_size=cfg.buffer_max_size)
    timer = Timer()

    print(
        f"training on {len(prompts)} prompts | "
        f"batch={cfg.n_prompts_per_step}x{cfg.group_size} | lr={cfg.lr}"
    )

    def run_one_step() -> tuple[float, float, dict, int, str, dict]:
        """One full training step.

        Returns (reward, acc, metrics, resp_tok, sample, prof) — `prof`
        carries the per-step Phase-4 numbers for the log line.
        """
        # TODO P4.2 (a): restart the CUDA high-water mark so this step's
        # peak reading is THIS step's maximum: torch.cuda.reset_peak_memory_stats()
        # (without it the reading is the max over all steps so far — the
        # exact bug the Phase-3 benchmark shipped with).
        torch.cuda.reset_peak_memory_stats()

        questions = random.sample(prompts, cfg.n_prompts_per_step)
        ground_truths = [
            r["answer"] for r in questions for _ in range(cfg.group_size)
        ]

        # ---- rollout (policy sampling, no gradients) ----
        with timer.section("rollout"):
            roll = rollout_worker.generate(
                [r["question"] for r in questions]
            )

        # ---- reward ----
        with timer.section("reward"):
            rewards = reward_mgr.compute(roll.responses, ground_truths)

        # ---- experience + buffer ----
        exp = Experience(
            prompt_ids=roll.prompt_ids,
            response_ids=roll.response_ids,
            attention_mask=roll.attention_mask,
            response_mask=Experience.build_response_mask(
                roll.attention_mask, roll.prompt_width
            ),
            rewards=rewards.to(cfg.device),
        )
        buffer.add(exp)

        # ---- training (consume everything buffered) ----
        buffered_all = buffer.get()
        with timer.section("prepare"):  # old + ref logprobs (no grads)
            for buffered in buffered_all:
                trainer.prepare(buffered)
        metrics: dict[str, float] = {}
        with timer.section("train"):  # forward + backward + optimizer
            for buffered in buffered_all:
                metrics = trainer.train_step(buffered)

        # TODO P4.2 (b): the three throughput numbers for the log line.
        #   rollout_tok_s = exp.n_response_tokens / timer.last("rollout")
        #     numerator = REAL response tokens (padding excluded — we do
        #     not bill for padding; Phase 5 measures how much we waste)
        #   train_tok_s   = same numerator / timer.last("train")
        #     the contrast is the point: rollout emits tokens ONE AT A
        #     TIME (autoregressive decode), training consumes them all
        #     in ONE parallel forward (teacher forcing)
        #   peak_gib      = torch.cuda.max_memory_allocated() / 1024**3
        #     meaningful per-step only because of the reset in (a)
        rollout_tok_s = exp.n_response_tokens / timer.last("rollout")
        # 因为我们训练也用这些token
        train_tok_s = exp.n_response_tokens / timer.last("train")
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3

        sample = roll.responses[0][:120]
        prof = {
            "rollout_tok_s": rollout_tok_s,
            "train_tok_s": train_tok_s,
            "peak_gib": peak_gib,
        }
        return (
            exp.rewards.mean().item(),
            (exp.rewards > 0).float().mean().item(),
            metrics,
            exp.n_response_tokens,
            sample,
            prof,
        )

    step = 0
    total_resp = 0
    while step < cfg.max_steps:
        step += 1
        t0 = time.perf_counter()
        try:
            reward, acc, metrics, resp_tok, sample, prof = run_one_step()
            total_resp += resp_tok
        except torch.OutOfMemoryError as e:
            # TODO P3.5: OOM diagnostics + batch-size fallback.
            #
            # 1. free the cached blocks: torch.cuda.empty_cache()
            # 2. print a diagnostic line: current step, n_prompts_per_step,
            #    group_size, max_new_tokens, and torch.cuda.max_memory_allocated()
            #    — show the user WHAT the memory pressure was
            # 3. shrink the rollout width: cfg.n_prompts_per_step = max(1,
            #    cfg.n_prompts_per_step - 1)  (samples per forward is
            #    n_prompts_per_step * group_size — the OOM driver)
            # 4. if it was already 1, re-raise: retrying the same failing
            #    config forever helps no one
            # 5. retry this step WITHOUT counting it as trained (step -= 1)
            buffer.get()
            torch.cuda.empty_cache()
            peak_mem = torch.cuda.max_memory_allocated()
            print(
                f"[OOM] step={step}, "
                f"n_prompts_per_step={cfg.n_prompts_per_step}, "
                f"group_size={cfg.group_size}, "
                f"max_new_tokens={cfg.max_new_tokens}, "
                f"peak_memory={peak_mem / 1024**3:.2f} GB"
            )
            if cfg.n_prompts_per_step == 1:
                raise 
            cfg.n_prompts_per_step = max(1, cfg.n_prompts_per_step-1)
            step -= 1
            continue


        # ---- logging ----
        print(
            f"step {step:3d} | reward {reward:.3f} "
            f"(acc {acc:.2f}) | kl {metrics.get('kl', float('nan')):.4f} "
            f"| clip {metrics.get('clip_frac', float('nan')):.3f} "
            f"| ratio [{metrics.get('ratio_min', float('nan')):.2f}, "
            f"{metrics.get('ratio_max', float('nan')):.2f}] "
            f"| grad {metrics.get('grad_norm', float('nan')):.2f} "
            f"| resp_tok {resp_tok} "
            f"| peak {prof['peak_gib']:.2f}G "
            f"| gen {prof['rollout_tok_s']:.0f} t/s "
            f"| trn {prof['train_tok_s']:.0f} t/s "
            f"| {time.perf_counter() - t0:.1f}s "
            f"(roll {timer.last('rollout'):.1f} / "
            f"prep {timer.last('prepare'):.1f} / "
            f"train {timer.last('train'):.1f})"
        )
        print(f"  sample response: {sample!r}")

    return {
        "timer": timer,
        "resp_tokens": total_resp,
        "steps": cfg.max_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for field, default in TrainingConfig.__dict__.items():
        if not field.startswith("_"):
            if isinstance(default, bool):
                # store_true: flag present -> True; absent -> default.
                # (type=bool would parse the STRING "False" as True!)
                parser.add_argument(
                    f"--{field.replace('_', '-')}",
                    action="store_true",
                    default=default,
                )
            else:
                parser.add_argument(
                    f"--{field.replace('_', '-')}",
                    type=type(default),
                    default=default,
                )
    parser.add_argument(
        "--toy", action="store_true", help="use the toy arithmetic task"
    )
    parser.add_argument(
        "--n-train", type=int, default=None, help="limit number of prompts"
    )
    args = parser.parse_args()

    cfg = TrainingConfig(
        **{k: v for k, v in vars(args).items() if k in TrainingConfig.__dict__}
    )
    prompts = make_toy_prompts() if args.toy else load_prompts(args.n_train)
    train(cfg, prompts)


if __name__ == "__main__":
    main()
