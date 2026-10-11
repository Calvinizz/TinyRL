"""Phase 6 — producer/consumer decoupling of rollout and training.

Same data path in both modes (queue -> stamp -> staleness gate -> train);
only WHO runs the producer differs:

  --mode sync    produce_one() and process() alternate in lockstep on
                 the main thread. The producer can never run ahead, so
                 every experience is consumed at the version it was
                 sampled under: lag == 0 by construction. This is the
                 Phase 2-5 loop, rerouted through a queue for a fair
                 comparison.

  --mode async   a producer thread pushes experiences up to queue_size
                 steps ahead while the main thread consumes and trains.
                 Rollout and training overlap; experiences may be stale;
                 the max-staleness gate drops the worst offenders.

Run (toy task):
    python scripts/train_async.py --toy --mode sync --max-steps 10
    python scripts/train_async.py --toy --mode async --max-steps 10

Two thread-safety notes that shape the code:

  * Timer is one-section-at-a-time (NOT thread-safe) — hence TWO Timer
    instances, one per thread (prod_timer / cons_timer), merged only in
    reporting. Making Timer itself thread-safe would need a lock per
    section or a thread-local _start; two instances is simpler.

  * PROTOTYPE HAZARD, accepted for the study: in async mode the
    consumer's optimizer.step() updates LoRA weights IN PLACE while the
    producer thread may be mid-generate() on the same module — kernels
    from both threads share the default CUDA stream and serialize, so
    nothing crashes, but one rollout can straddle two policy versions.
    The stamp is taken BEFORE generate(), so a straddled rollout is
    recorded with the OLDER version and its measured lag is a lower
    bound. Production systems close this hole with atomic weight sync
    (versioned broadcast to a rollout replica); we measure the hole
    instead of fixing it — see benchmarks/decoupling.md.

No OOM fallback here (prototype): run with the batch size you verified
in train_tinyrl.py first.
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyr.config import TrainingConfig
from tinyr.experience import Experience, ExperienceQueue
from tinyr.models import ModelManager
from tinyr.profiling import Timer
from tinyr.reward import RewardManager
from tinyr.rollout import RolloutWorker
from tinyr.trainer import GRPOTrainer, PolicyVersion, policy_lag


def sync_weights(backend, policy, version: int) -> None:
    """Ship the trainer's current LoRA to the engine (P7.3 (c)).

    Per-version directory (lora_cache/v{N}, gitignored): one fixed
    path would race the engine's adapter cache. ~9 MiB per step —
    prune old versions when convenient; the newest doubles as an
    eval checkpoint. Called AFTER version.bump() so name, path and
    stamp agree; experiences produced before the sync lands are
    stamped from backend.served_version (the old one) — honest lag,
    exactly what the staleness gate is for.
    """
    cache = Path(__file__).resolve().parents[1] / "lora_cache"
    d = cache / f"v{version}"
    policy.save_pretrained(d)      # adapter_config.json + .safetensors
    backend.sync_policy(version, str(d))


def train(cfg: TrainingConfig, prompts: list, mode: str) -> dict:
    """Run the pipeline in `mode`; returns stats for the benchmark."""
    assert mode in ("sync", "async")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    model_mgr = ModelManager(cfg)
    if cfg.rollout_backend == "vllm":
        from tinyr.config import MODEL_PATH
        from tinyr.rollout.vllm_backend import VLLMRolloutBackend

        backend = VLLMRolloutBackend(
            str(MODEL_PATH), model_mgr.tokenizer, cfg
        )
    else:
        backend = None  # RolloutWorker defaults to HFRolloutBackend
    rollout_worker = RolloutWorker(
        model_mgr.policy, model_mgr.tokenizer, cfg, backend=backend
    )
    reward_mgr = RewardManager()
    trainer = GRPOTrainer(model_mgr.policy, model_mgr.ref, cfg)

    version = PolicyVersion()
    queue = ExperienceQueue(cfg.queue_size)
    prod_timer = Timer()   # producer thread's sections
    cons_timer = Timer()   # consumer (main) thread's sections

    stats = {"steps": 0, "drops": 0, "lags": [], "resp_tokens": 0}
    torch.cuda.reset_peak_memory_stats(torch.device(cfg.device))

    # ---- producer side ------------------------------------------------

    def produce_one() -> Experience:
        """Sample one batch and stamp it with the version it came from."""
        questions = random.sample(prompts, cfg.n_prompts_per_step)
        ground_truths = [
            r["answer"] for r in questions for _ in range(cfg.group_size)
        ]
        # stamp BEFORE generate(): the weights being sampled from are
        # version `v` right now. P7.3 sharpened this: a dedicated
        # engine lags the trainer between sync_policy() calls, and
        # stamping the trainer's CURRENT version would understate lag —
        # stamp what the backend is actually SERVING (None = it shares
        # the policy object, so "current" is the honest stamp).
        v = rollout_worker.backend.served_version
        if v is None:
            v = version.current
        with prod_timer.section("rollout"):
            roll = rollout_worker.generate([r["question"] for r in questions])
        with prod_timer.section("reward"):
            rewards = reward_mgr.compute(roll.responses, ground_truths)
        return Experience(
            prompt_ids=roll.prompt_ids,
            response_ids=roll.response_ids,
            attention_mask=roll.attention_mask,
            response_mask=Experience.build_response_mask(
                roll.attention_mask, roll.prompt_width
            ),
            rewards=rewards.to(cfg.device),
            policy_version=v,
        )

    def producer_main() -> None:
        for _ in range(cfg.max_steps):
            queue.put(produce_one())
        queue.put(None)  # poison pill: "no more work" — consumer exits

    # ---- consumer side ------------------------------------------------

    def process(exp: Experience) -> None:
        """Gate one experience on staleness, then prepare + train + bump."""
        # ---- TODO P6.4: the staleness gate ---------------------------
        # lag = policy_lag(version.current, exp.policy_version)
        #   -> the trainer has moved `lag` optimizer steps past the
        #      policy this batch was sampled under.
        # stats["lags"].append(lag)
        # if lag > cfg.max_staleness:
        #     too stale to train on — count it in stats["drops"], print
        #     a drop line, and RETURN (skip prepare/train entirely).
        #     In sync mode this never fires (lag == 0 by construction);
        #     in async it is the valve that bounds off-policy drift.
        # ----------------------------------------------------------------
        lag = policy_lag(version.current,exp.policy_version)
        stats["lags"].append(lag)
        if lag > cfg.max_staleness:
            stats["drops"] += 1
            print(
                f"drop | lag {lag} > max_staleness {cfg.max_staleness} "
                f"| sampled v{exp.policy_version} | current v{version.current}"
            )
            return None

        with cons_timer.section("prepare"):  # old + ref logprobs (no grads)
            trainer.prepare(exp)
        with cons_timer.section("train"):    # forward + backward + step
            metrics = trainer.train_step(exp)
        version.bump()  # one optimizer step = one version
        if cfg.rollout_backend == "vllm":
            sync_weights(rollout_worker.backend, model_mgr.policy,
                         version.current)

        stats["steps"] += 1
        stats["resp_tokens"] += exp.n_response_tokens
        print(
            f"step {stats['steps']:3d} | lag {lag} "
            f"| reward {exp.rewards.mean().item():.3f} "
            f"| kl {metrics.get('kl', float('nan')):.4f} "
            f"| clip {metrics.get('clip_frac', float('nan')):.3f} "
            f"| q {len(queue)}/{cfg.queue_size}"
        )

    def consumer_main() -> None:
        while True:
            exp = queue.get()
            if exp is None:
                break
            process(exp)

    # ---- orchestration -------------------------------------------------

    print(
        f"mode {mode} | batch={cfg.n_prompts_per_step}x{cfg.group_size} "
        f"| queue {cfg.queue_size} | max_staleness {cfg.max_staleness} "
        f"| steps {cfg.max_steps}"
    )
    wall_start = time.perf_counter()

    if mode == "async":
        producer = threading.Thread(target=producer_main)
        producer.start()
        consumer_main()  # exits on the poison pill
        producer.join()
    else:
        # strict lockstep: produce one, consume one — the queue is
        # traversed (same code path) but never holds a spare item
        for _ in range(cfg.max_steps):
            queue.put(produce_one())
            process(queue.get())

    wall = time.perf_counter() - wall_start
    lags = stats["lags"]
    mean_lag = sum(lags) / len(lags) if lags else 0.0
    # peak of the TRAINER's device: with rollout on its own GPU
    # (cfg.device="cuda:1"), the default-device query would report the
    # engine's card instead — in THIS process the trainer is the only
    # interesting allocator
    dev = torch.device(cfg.device)
    peak_gib = torch.cuda.max_memory_allocated(dev) / 1024**3
    print(
        f"[{mode}] trained {stats['steps']} | dropped {stats['drops']} "
        f"| lag mean {mean_lag:.2f} max {max(lags, default=0)} "
        f"| wall {wall:.1f}s | {stats['resp_tokens']} resp tok "
        f"| peak {peak_gib:.2f} GiB on {cfg.device}"
    )
    return {
        **stats,
        "wall": wall,
        "peak_gib": peak_gib,
        "prod_timer": prod_timer,
        "cons_timer": cons_timer,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for field, default in TrainingConfig.__dict__.items():
        if not field.startswith("_"):
            if isinstance(default, bool):
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
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--mode", choices=("sync", "async"), default="sync")
    args = parser.parse_args()

    cfg = TrainingConfig(
        **{k: v for k, v in vars(args).items() if k in TrainingConfig.__dict__}
    )
    if args.toy:
        from scripts.train_tinyrl import make_toy_prompts

        prompts = make_toy_prompts()
    else:
        from scripts.train_tinyrl import load_prompts

        prompts = load_prompts(None)

    train(cfg, prompts, args.mode)


if __name__ == "__main__":
    main()
