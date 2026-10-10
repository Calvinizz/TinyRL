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


def train(cfg: TrainingConfig, prompts: list[dict]) -> None:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    model_mgr = ModelManager(cfg)
    rollout_worker = RolloutWorker(model_mgr.policy, model_mgr.tokenizer, cfg)
    reward_mgr = RewardManager()
    trainer = GRPOTrainer(model_mgr.policy, model_mgr.ref, cfg)
    buffer = ExperienceBuffer(max_size=cfg.buffer_max_size)

    print(
        f"training on {len(prompts)} prompts | "
        f"batch={cfg.n_prompts_per_step}x{cfg.group_size} | lr={cfg.lr}"
    )

    for step in range(1, cfg.max_steps + 1):
        t0 = time.perf_counter()
        questions = random.sample(prompts, cfg.n_prompts_per_step)
        ground_truths = [
            r["answer"] for r in questions for _ in range(cfg.group_size)
        ]

        # ---- rollout (policy sampling, no gradients) ----
        roll = rollout_worker.generate([r["question"] for r in questions])

        # ---- reward ----
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
        metrics: dict[str, float] = {}
        for buffered in buffer.get():
            trainer.prepare(buffered)
            metrics = trainer.train_step(buffered)

        # ---- logging ----
        acc = (exp.rewards > 0).float().mean().item()
        print(
            f"step {step:3d} | reward {exp.rewards.mean().item():.3f} "
            f"(acc {acc:.2f}) | kl {metrics.get('kl', float('nan')):.4f} "
            f"| clip {metrics.get('clip_frac', float('nan')):.3f} "
            f"| ratio [{metrics.get('ratio_min', float('nan')):.2f}, "
            f"{metrics.get('ratio_max', float('nan')):.2f}] "
            f"| grad {metrics.get('grad_norm', float('nan')):.2f} "
            f"| resp_tok {exp.n_response_tokens} "
            f"| {time.perf_counter() - t0:.1f}s"
        )
        print(f"  sample response: {roll.responses[0][:120]!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    for field, default in TrainingConfig.__dict__.items():
        if not field.startswith("_"):
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
