"""P8.2 — engine + DDP trainers: the full distributed GRPO step.

Topology (sidecar, study §7 option B — chosen because vLLM's own
distributed init inside a torchrun world is a version-dependent
minefield we have no need to walk at 0.5B scale):

    python scripts/train_ddp.py             (parent: no CUDA, babysits)
      |- engine  subprocess, CUDA_VISIBLE_DEVICES=0
      |    vLLM + reward; produces experiences until max_steps pushed
      `- torchrun --nproc_per_node=N, CUDA_VISIBLE_DEVICES=1,2,..
           `- trainer ranks (DDP world); rank 0 bridges the queues

Data path — everything crosses ranks exactly once, at one point:

    engine:  generate -> reward -> stamp served_version -> exp_q (CPU)
    rank 0:  exp_q.get() -> dist.broadcast_object_list
    ranks:   staleness gate (IDENTICAL verdicts: same exp, same version
             counter — a divergent drop desyncs backwards and hangs the
             world) -> prepare -> train_step (AllReduce in backward)
    rank 0:  save LoRA -> cmd_q -> engine applies sync_policy

Two exactness details single-GPU training never faces:
  * loss weighting — DDP averages per-rank losses, but a token-mean per
    shard weights ranks by 1/world regardless of token counts; the
    exact global mean needs scale_r = world*tok_r/total (one scalar
    allreduce per step), applied via trainer.loss_scale.
  * group integrity — shards are cut at the PROMPT level (advantages
    are intra-group statistics; a split group is two fragments of a
    number neither rank can finish). group_rows_for_rank.

Run (server, 3 GPUs — engine on 0, two DDP ranks on 1,2):
    cd /data/zy/TinyRL && source scripts/server_env.sh
    python scripts/train_ddp.py --max-steps 10

Protocol matches benchmarks/rematch.md legs (batch 4x2, queue 2,
max_staleness 3) so the wall/step number is directly comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import queue as queue_mod
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyr.config import MODEL_PATH, TrainingConfig  # noqa: E402
from tinyr.trainer.distributed import (  # noqa: E402
    ddp_loss_scale,
    even_len,
    group_rows_for_rank,
)

REPO = Path(__file__).resolve().parents[1]

# parent -> children rendezvous, passed through the environment (the
# children are unrelated processes: subprocess + torchrun workers, so
# in-process mp.Queue handles cannot travel — a remote manager can)
ENV_PORT = "TINYRL_Q_PORT"
ENV_AUTH = "TINYRL_Q_AUTH"
ENV_RESULT = "TINYRL_RESULT"
ENV_MAX_STEPS = "TINYRL_MAX_STEPS"


# ------------------------------------------------------------------- IPC
def _manager_class():
    from multiprocessing.managers import BaseManager

    class _Queues(BaseManager):
        pass

    return _Queues


def host_queues(port: int, authkey: bytes, queue_size: int):
    """Parent side: serve exp/cmd queues through a remote manager.

    queue.Queue (thread queue), NOT mp.Queue: the manager is a threaded
    server, and a plain queue travels as pure pickled bytes over the
    manager socket. mp.Queue carries Pipes/Connections whose auto-
    proxying detours through fd-passing (resource_sharer) — and the
    authkey challenge there uses each process's own key, which blew up
    the first run with AuthenticationError: digest sent was rejected.
    """
    exp_q = queue_mod.Queue(maxsize=queue_size)
    cmd_q = queue_mod.Queue()
    Q = _manager_class()
    Q.register("exp_q", callable=lambda: exp_q)
    Q.register("cmd_q", callable=lambda: cmd_q)
    mgr = Q(address=("127.0.0.1", port), authkey=authkey)
    mgr.start()  # the manager server is its own process; proxies below
    return mgr, mgr.exp_q(), mgr.cmd_q()


def connect_queues(port: int, authkey: bytes):
    """Child side (engine process or trainer rank 0): join the manager."""
    Q = _manager_class()
    Q.register("exp_q")
    Q.register("cmd_q")
    mgr = Q(address=("127.0.0.1", port), authkey=authkey)
    mgr.connect()
    return mgr.exp_q(), mgr.cmd_q()


# ---------------------------------------------------------------- parent
def run_parent(args) -> None:
    port = 29600 + os.getpid() % 500
    authkey = f"tinyrl-{os.getpid()}".encode()
    result = REPO / "lora_cache" / "ddp_result.json"

    cfg = TrainingConfig()
    mgr, _, _ = host_queues(port, authkey, cfg.queue_size)

    here = str(Path(__file__).resolve())
    common = dict(os.environ)
    # hermetic engine startup — same pins, same reason, as
    # tests/test_backend.py: flashinfer's JIT sampler nvcc-chokes on
    # this box, and fork inherits a threaded parent that deadlocks
    # vLLM's EngineCore. setdefault: shell values still win.
    common.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    common.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    common.update(
        {
            ENV_PORT: str(port),
            ENV_AUTH: authkey.decode(),
            ENV_RESULT: str(result),
            ENV_MAX_STEPS: str(args.max_steps),
        }
    )
    torchrun = shutil.which("torchrun") or str(
        Path(sys.executable).parent / "torchrun"
    )
    procs = {
        "engine": subprocess.Popen(
            [sys.executable, here, "--role", "engine"],
            env={**common, "CUDA_VISIBLE_DEVICES": str(args.engine_gpu)},
            cwd=str(REPO),
        ),
        "trainers": subprocess.Popen(
            [
                torchrun,
                f"--nproc_per_node={args.world}",
                here,
                "--role",
                "trainer",
            ],
            env={**common, "CUDA_VISIBLE_DEVICES": args.trainer_gpus},
            cwd=str(REPO),
        ),
    }
    print(
        f"[parent] engine on GPU {args.engine_gpu}, {args.world} DDP ranks "
        f"on GPUs {args.trainer_gpus} | {args.max_steps} steps"
    )

    deadline = time.time() + args.timeout
    try:
        for name, p in procs.items():
            remaining = max(1.0, deadline - time.time())
            try:
                p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                print(f"[parent] TIMEOUT waiting for {name} — killing the tree")
                for q in procs.values():
                    q.terminate()
                sys.exit(1)
    finally:
        mgr.shutdown()

    if procs["trainers"].returncode == 0 and result.exists():
        stats = json.loads(result.read_text())
        print("DDP_JSON " + json.dumps(stats))
    else:
        print(
            f"[parent] engine rc={procs['engine'].returncode} "
            f"trainers rc={procs['trainers'].returncode}"
        )
        sys.exit(1)


# ---------------------------------------------------------------- engine
def _drain_sync(cmd_q, backend) -> int:
    """Apply every pending weight-sync command; return the last version."""
    v = None
    while True:
        try:
            v, path = cmd_q.get_nowait()
        except queue_mod.Empty:
            break
        backend.sync_policy(v, path)
    return v


def run_engine() -> None:
    port, authkey = int(os.environ[ENV_PORT]), os.environ[ENV_AUTH].encode()
    max_steps = int(os.environ[ENV_MAX_STEPS])
    # cpu keeps every Experience tensor CPU-side: the manager pickles
    # them through a socket, and CUDA tensors do not survive that
    cfg = TrainingConfig(rollout_backend="vllm", device="cpu")

    exp_q, cmd_q = connect_queues(port, authkey)
    from transformers import AutoTokenizer

    from scripts.train_tinyrl import make_toy_prompts
    from tinyr.experience import Experience
    from tinyr.reward import RewardManager
    from tinyr.rollout import RolloutWorker
    from tinyr.rollout.vllm_backend import VLLMRolloutBackend

    random.seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    backend = VLLMRolloutBackend(str(MODEL_PATH), tokenizer, cfg)
    worker = RolloutWorker(None, tokenizer, cfg, backend=backend)
    reward_mgr = RewardManager()
    prompts = make_toy_prompts()
    print(f"[engine] up; base weights = version 0 (fresh LoRA: lora_B == 0)")

    for step in range(max_steps):
        t0 = time.perf_counter()
        _drain_sync(cmd_q, backend)  # newest weights BEFORE sampling
        t_sync = time.perf_counter() - t0
        t0 = time.perf_counter()
        questions = random.sample(prompts, cfg.n_prompts_per_step)
        truths = [r["answer"] for r in questions for _ in range(cfg.group_size)]
        # stamp what is actually being SERVED: None before any sync means
        # base/zero-adapter behavior == version 0 in the trainer's space
        v = backend.served_version if backend.served_version is not None else 0
        roll = worker.generate([r["question"] for r in questions])
        rewards = reward_mgr.compute(roll.responses, truths)
        exp = Experience(
            prompt_ids=roll.prompt_ids,
            response_ids=roll.response_ids,
            attention_mask=roll.attention_mask,
            response_mask=Experience.build_response_mask(
                roll.attention_mask, roll.prompt_width
            ),
            rewards=rewards,
            policy_version=v,
        )
        # numpy at the boundary — see exp_to_payload
        exp_q.put(exp_to_payload(exp))  # blocks when full: Phase-6 backpressure
        t_gen = time.perf_counter() - t0
        print(f"[engine] step {step + 1}/{max_steps} stamped v{v} "
              f"| gen+reward {t_gen:.2f}s | sync {t_sync:.2f}s")
    print(f"[engine] produced {max_steps} experiences; final served v{backend.served_version}")


# --------------------------------------------------------------- trainer
def exp_to_payload(exp) -> dict:
    """Experience -> plain numpy dict for the queue crossing.

    torch installs custom pickle reducers on tensors: cross-process
    pickling detours through shared-memory fd passing (resource_sharer),
    whose authkey challenge only works within one process tree — an
    unrelated manager server dies with 'digest sent was rejected'.
    numpy arrays pickle as plain bytes: no magic, no authkey. Tensors
    are rebuilt on the trainer's own device afterwards.
    """
    return {
        "prompt_ids": exp.prompt_ids.numpy(),
        "response_ids": exp.response_ids.numpy(),
        "attention_mask": exp.attention_mask.numpy(),
        "response_mask": exp.response_mask.numpy(),
        "rewards": exp.rewards.numpy(),
        "policy_version": exp.policy_version,
    }


def payload_to_experience(p: dict, device: torch.device):
    from tinyr.experience import Experience

    return Experience(
        prompt_ids=torch.from_numpy(p["prompt_ids"]).to(device),
        response_ids=torch.from_numpy(p["response_ids"]).to(device),
        attention_mask=torch.from_numpy(p["attention_mask"]).to(device),
        response_mask=torch.from_numpy(p["response_mask"]).to(device),
        rewards=torch.from_numpy(p["rewards"]).to(device),
        policy_version=p["policy_version"],
    )


def sync_weights(cmd_q, policy, version: int) -> None:
    """Rank 0's replica IS the agreed post-AllReduce average — only it
    ships weights (P7.3 machinery: versioned dir + engine hot-swap)."""
    d = REPO / "lora_cache" / f"v{version}"
    policy.save_pretrained(d)
    cmd_q.put((version, str(d)))


def run_trainer() -> None:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    # P8.1 (a) verbatim: join the NCCL world
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")

    port, authkey = int(os.environ[ENV_PORT]), os.environ[ENV_AUTH].encode()
    max_steps, result_path = int(os.environ[ENV_MAX_STEPS]), os.environ[ENV_RESULT]

    from tinyr.models import ModelManager
    from tinyr.trainer import GRPOTrainer, PolicyVersion, policy_lag

    cfg = TrainingConfig(device=str(device))
    torch.manual_seed(cfg.seed)  # identical replicas, every rank
    model_mgr = ModelManager(cfg)
    policy = model_mgr.policy
    ddp = DistributedDataParallel(policy, device_ids=[local_rank])
    trainer = GRPOTrainer(ddp, model_mgr.ref, cfg)

    # only rank 0 touches the queue — the broadcast IS the boundary
    exp_q = cmd_q = None
    if rank == 0:
        exp_q, cmd_q = connect_queues(port, authkey)

    version = PolicyVersion()
    stats = {"steps": 0, "drops": 0, "lags": [], "resp_tokens": 0}
    torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.perf_counter()

    for i in range(max_steps):
        # ---- the boundary: one queue reader, one broadcast ------------
        t0 = time.perf_counter()
        payload = [None]
        if rank == 0:
            payload[0] = exp_q.get()  # blocks: backpressure reaches the engine
        dist.broadcast_object_list(payload, src=0)
        # plain bytes over the broadcast; tensors are born on THIS rank's GPU
        exp = payload_to_experience(payload[0], device)
        t_bcast = time.perf_counter() - t0

        # ---- staleness gate: identical verdicts on every rank ---------
        lag = policy_lag(version.current, exp.policy_version)
        stats["lags"].append(lag)
        if lag > cfg.max_staleness:
            stats["drops"] += 1
            if rank == 0:
                print(f"drop | lag {lag} > {cfg.max_staleness} | sampled v{exp.policy_version}")
            continue

        # ---- exact global token mean under DDP averaging --------------
        t0 = time.perf_counter()
        tok = float(exp.response_mask[:, 1:].sum())
        t = torch.tensor([tok], device=device)
        dist.all_reduce(t)
        trainer.loss_scale = ddp_loss_scale(tok, t.item(), world)

        # ---- prompt-level shard: whole groups per rank ----------------
        n_prompts = even_len(cfg.n_prompts_per_step, world)
        shard = exp.rows(group_rows_for_rank(n_prompts, cfg.group_size, rank, world))

        trainer.prepare(shard)   # advantages (intra-group) + old/ref logprobs
        t_prep = time.perf_counter() - t0

        t0 = time.perf_counter()
        metrics = trainer.train_step(shard)  # AllReduce fires in backward
        t_train = time.perf_counter() - t0

        t0 = time.perf_counter()
        version.bump()
        stats["steps"] += 1
        stats["resp_tokens"] += exp.n_response_tokens
        if rank == 0:
            sync_weights(cmd_q, ddp.module, version.current)
        dist.barrier()  # timing fence: no rank sneaks into the next get
        t_sync = time.perf_counter() - t0
        stats.setdefault("phases", []).append(
            {"bcast": round(t_bcast, 3), "prep": round(t_prep, 3),
             "train": round(t_train, 3), "sync": round(t_sync, 3)}
        )
        if rank == 0:
            print(
                f"step {stats['steps']:3d} | lag {lag} "
                f"| reward {exp.rewards.mean().item():.3f} "
                f"| kl {metrics.get('kl', float('nan')):.4f} "
                f"| clip {metrics.get('clip_frac', float('nan')):.3f} "
                f"| bcast {t_bcast:.2f} prep {t_prep:.2f} "
                f"train {t_train:.2f} sync {t_sync:.2f}"
            )

    wall = time.perf_counter() - wall_start
    lags = stats["lags"]
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    peaks = [None] * world
    dist.all_gather_object(peaks, round(peak, 2))
    if rank == 0:
        ph = stats.get("phases") or []
        phase_means = {k: round(sum(p[k] for p in ph) / len(ph), 3) for k in ph[0]} if ph else {}
        out = {
            "steps": stats["steps"],
            "drops": stats["drops"],
            "lag_mean": round(sum(lags) / len(lags), 2) if lags else 0.0,
            "lag_max": max(lags, default=0),
            "wall_s": round(wall, 2),
            "resp_tokens": stats["resp_tokens"],
            "resp_tok_s": round(stats["resp_tokens"] / wall, 1),
            "world": world,
            "peaks_gib": peaks,
            "phase_means_s": phase_means,
        }
        Path(result_path).parent.mkdir(parents=True, exist_ok=True)
        Path(result_path).write_text(json.dumps(out))
        print(
            f"[trainer] trained {stats['steps']} | dropped {stats['drops']} "
            f"| lag mean {out['lag_mean']} max {out['lag_max']} "
            f"| wall {wall:.1f}s | peaks {peaks} GiB"
        )
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("parent", "engine", "trainer"), default="parent")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--world", type=int, default=2)
    parser.add_argument("--engine-gpu", type=int, default=0)
    parser.add_argument("--trainer-gpus", type=str, default="1,2")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    if args.role == "parent":
        run_parent(args)
    elif args.role == "engine":
        run_engine()
    else:
        run_trainer()


if __name__ == "__main__":
    main()
