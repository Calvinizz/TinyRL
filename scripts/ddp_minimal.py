"""P8.1 — the minimal DDP example: what gradient AllReduce actually is.

Three claims, seen instead of trusted:

  1. DDP's gradient == the AVERAGE of every rank's local gradient.
     That average IS the entire synchronization semantics — nothing
     else moves between ranks.
  2. After backward, every rank holds the SAME gradient (that is what
     AllReduce means: reduce, then the result is everywhere).
  3. Every rank runs its OWN optimizer on its OWN replica — nobody is
     "in charge". Same average + same optimizer state + deterministic
     AdamW = replicas stay identical forever. The agreement is the
     average, not a master.

The loss is deliberately NOT RL — `logits.mean()` on a fixed synthetic
batch. Phase 8 wants the distributed primitives in isolation first;
the RL wiring is P8.2.

Run on the server (3 free GPUs, real NCCL):
    cd /data/zy/TinyRL && source scripts/server_env.sh
    /data/zy/tinyrl390/bin/torchrun --nproc_per_node=3 scripts/ddp_minimal.py

Local sanity (single rank — AllReduce over a world of one; two ranks
on one GPU is NOT allowed under NCCL, "duplicate GPU"):
    torchrun --nproc_per_node=1 scripts/ddp_minimal.py

Acceptance: every rank prints three True VERDICT lines.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyr.config import MODEL_PATH, TrainingConfig  # noqa: E402
from tinyr.trainer.distributed import even_len, shard_for_rank  # noqa: E402

N_ROWS = 12   # divisible by world sizes 1/2/3/4 — and even_len guards anyway
N_TOKENS = 32


# --------------------------------------------------------------------------
# TODO P8.1 (a) — join the process group
# --------------------------------------------------------------------------
def setup_distributed() -> tuple[int, int, int, torch.device]:
    """Read torchrun's environment and join the NCCL world.

    torchrun exports RANK / LOCAL_RANK / WORLD_SIZE (and MASTER_ADDR /
    MASTER_PORT) into the environment of every worker; you do NOT pass
    them by hand.

    Return (rank, local_rank, world_size, device).

    The order that avoids the classic startup errors:
      1. rank, local_rank, world_size = int(os.environ[...]), ...
      2. device = f"cuda:{local_rank}"            # one GPU per rank;
         torch.cuda.set_device(local_rank)        # NCCL init binds to the
                                                  # CURRENT device
      3. dist.init_process_group("nccl")          # backend reads the
                                                  # MASTER/.rank env alone

    Pitfalls worth hitting once on purpose:
      - "cuda:0" hardcoded on every rank -> duplicate-GPU NCCL error
      - cuda.set_device AFTER init_process_group -> rank/device mismatch
      - gloo instead of nccl also works (CPU debugging, no driver)
    """
    raise NotImplementedError("P8.1 (a)")


# --------------------------------------------------------------------------
# scaffold — identical replicas, fixed batch, strided shards
# --------------------------------------------------------------------------
def build_policy(cfg: TrainingConfig, device: torch.device):
    """Same weights on EVERY rank: same seed + same construction order.

    Fresh LoRA => lora_B == 0 (the P7.3 zero-adapter identity again),
    which also means lora_A gets ZERO gradient here: dL/dA = B^T(...) = 0
    by the chain rule. Probe lora_B tensors, not lora_A.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(cfg.seed)
    base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16)
    policy = get_peft_model(
        base,
        LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=0.0,  # determinism: two backwards must see one loss
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    ).to(device)
    return policy.eval()


def synthetic_batch(policy, cfg: TrainingConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed token matrix, identical on every rank (CPU generator)."""
    vocab = policy.get_input_embeddings().weight.shape[0]
    g = torch.Generator().manual_seed(cfg.seed + 1)
    ids = torch.randint(0, vocab, (N_ROWS, N_TOKENS), generator=g)
    return ids, torch.ones_like(ids)


def local_loss(module, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    """Deterministic differentiable loss — .float() so fp16 logits reduce
    in fp32 and the later comparisons have headroom."""
    logits = module(input_ids=ids, attention_mask=attn).logits
    return logits.float().mean()


# --------------------------------------------------------------------------
# TODO P8.1 (b) — the DDP wrapper
# --------------------------------------------------------------------------
def wrap_ddp(policy, device: torch.device):
    """Wrap the WHOLE peft model; return the DistributedDataParallel.

      ddp = DistributedDataParallel(policy, device_ids=[int(str(device).split(":")[1])])

    - wrap the peft model (outermost), not the base inside it
    - find_unused_parameters=False is correct here AND much faster: every
      LoRA tensor participates in forward (lora_A having a ZERO gradient
      is not the same as being UNUSED — unused means absent from the
      autograd graph; zero is a value the reducer happily reduces)
    - ddp.module IS policy (the wrapper is a thin proxy); keep both
      references, (c) needs the raw one for the optimizer
    """
    raise NotImplementedError("P8.1 (b)")


# --------------------------------------------------------------------------
# TODO P8.1 (c) — prove the three claims ("梯度在哪合并、谁说了算")
# --------------------------------------------------------------------------
def verify_sync_semantics(ddp, policy, ids_r, attn_r, world_size: int) -> dict[str, bool]:
    """Run the DDP backward, then verify what AllReduce actually did.

    Recipe (probe tensors: pick one lora_B weight; lora_A grads are 0):

    claim 1 — DDP grad == mean of local grads:
      1. loss = local_loss(ddp, ids_r, attn_r); loss.backward()
         -> grads are now the world-average (hooks fired AllReduce
            DURING backward — before optimizer.step, that is where
            gradients merge)
      2. snapshot = p.grad.detach().clone()
      3. policy.zero_grad()
      4. with ddp.no_sync():                       # gradients stay LOCAL
             local_loss(ddp, ids_r, attn_r).backward()
         PIT: do NOT backward through ddp.module directly to get local
         grads — the reducer's hooks live on the PARAMETERS and will
         raise "Expected to have finished reduction in the prior
         iteration". no_sync() is the sanctioned local mode (it is also
         how gradient accumulation skips intermediate allreduces).
      5. manual = p.grad.detach().clone()
         dist.all_reduce(manual)                   # SUM is the default op
         manual /= world_size
      6. claim1 = torch.allclose(manual.float(), snapshot.float(),
                                 atol=1e-5, rtol=1e-4)
         (two backward passes -> bitwise is too strict a promise;
          allclose with margin is the honest check)

    claim 2 — same gradient on every rank:
      checksum the SNAPSHOT (it is the synced grad), dist.all_gather,
      compare with torch.equal.

    claim 3 — replicas stay identical through optimizer.step():
      build AdamW over the trainable params (the GRPOTrainer pattern:
      [p for p in policy.parameters() if p.requires_grad] — every rank
      builds its OWN, nobody is in charge), step it, checksum a lora_B
      WEIGHT (it must have moved), all_gather, torch.equal.

    Return {"ddp_eq_manual_avg": ..., "grad_same_across_ranks": ...,
            "replicas_identical_after_step": ...}.
    """
    raise NotImplementedError("P8.1 (c)")


def main() -> None:
    rank, local_rank, world_size, device = setup_distributed()  # (a)

    cfg = TrainingConfig(device=str(device))
    policy = build_policy(cfg, device)
    ddp = wrap_ddp(policy, device)                              # (b)

    ids, attn = synthetic_batch(policy, cfg)
    rows = shard_for_rank(range(even_len(N_ROWS, world_size)), rank, world_size)
    ids_r, attn_r = ids[rows].to(device), attn[rows].to(device)  # ranks DISAGREE here

    results = verify_sync_semantics(ddp, policy, ids_r, attn_r, world_size)  # (c)

    print(f"[rank {rank}] rows {len(rows)}/{N_ROWS}  "
          + "  ".join(f"{k}={v}" for k, v in results.items()))
    if not all(results.values()):
        print(f"[rank {rank}] VERDICT FAIL — a claim above is False")

    import torch.distributed as dist

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":  # torchrun re-imports __main__ per worker (P7 lesson)
    main()
