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

The loss is deliberately NOT RL — next-token cross-entropy on a fixed
synthetic batch (the two naive losses die in opposite fp16 directions;
see local_loss). Phase 8 wants the distributed primitives in isolation
first; the RL wiring is P8.2.

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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyr.config import MODEL_PATH, TrainingConfig  # noqa: E402
from tinyr.trainer.distributed import even_len, shard_for_rank  # noqa: E402

N_ROWS = 12   # divisible by world sizes 1/2/3/4 — and even_len guards anyway
N_TOKENS = 32


# --------------------------------------------------------------------------
# P8.1 (a) — join the process group (walkthrough version)
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
    rank = int(os.environ["RANK"])             # global: unique across ALL nodes
    local_rank = int(os.environ["LOCAL_RANK"]) # per-node: which GPU on THIS box
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)              # BEFORE init: NCCL binds to the
                                               # current device when it builds
                                               # its communicators
    dist.init_process_group("nccl")            # init_method="env://" is the
                                               # default: MASTER_ADDR/PORT +
                                               # RANK/WORLD_SIZE, all from
                                               # torchrun. Blocks until every
                                               # rank has rendezvoused — even
                                               # init is a collective.
    return rank, local_rank, world_size, device


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
    """Standard LM loss on the synthetic text: predict token t+1.

    Cross-entropy, NOT logits.mean()/sum(): CE's per-logit gradient is
    softmax - onehot, bounded in [-1, 1] — naturally inside fp16's
    window. The naive losses both die, for opposite reasons (receipts
    in git history):
      mean over [N, T, V] ~ 1.9e7 elements -> per-logit grad ~5e-8,
        BELOW fp16's smallest subnormal (6e-8): underflow to exact
        zero through the 24 fp16 layers (printed grad|max|=0.0)
      sum (or x1e2) -> intermediates accumulate over the 151936-wide
        vocab dim past fp16's 65504 ceiling: inf -> nan
    This tightrope is why mixed-precision training carries a dynamic
    loss scaler (torch.amp.GradScaler) — and why real LM losses are
    means over ~1e2 tokens, not ~1e7 logits."""
    logits = module(input_ids=ids, attention_mask=attn).logits[:, :-1]
    targets = ids[:, 1:]
    return torch.nn.functional.cross_entropy(
        logits.float().flatten(0, 1), targets.flatten()
    )


# --------------------------------------------------------------------------
# P8.1 (b) — the DDP wrapper (user)
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
    return DistributedDataParallel(policy, device_ids=[int(str(device).split(":")[1])])



# --------------------------------------------------------------------------
# P8.1 (c) — prove the three claims ("梯度在哪合并、谁说了算")
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
    rank = dist.get_rank()  # the world knows who it is

    # The probe must be the SAME tensor on every rank. Construction is
    # seeded, so named_parameters() yields the same order everywhere;
    # a rank-dependent probe choice would fake a claim-2 failure.
    probe_name, probe = next(
        (n, p) for n, p in policy.named_parameters()
        if p.requires_grad and "lora_B" in n
    )

    def max_rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
        d = (a.float() - b.float()).abs().max().item()
        s = max(a.float().abs().max().item(), b.float().abs().max().item(), 1e-12)
        return d / s

    # ---- claim 1: DDP's grad == the AVERAGE of local grads ----------
    local_loss(ddp, ids_r, attn_r).backward()  # hooks fire AllReduce DURING
    snapshot = probe.grad.detach().clone()     # backward: this IS where grads
                                               # merge — before any step()
    policy.zero_grad()
    with ddp.no_sync():                       # same loss, gradients stay LOCAL
        local_loss(ddp, ids_r, attn_r).backward()
    manual = probe.grad.detach().clone()
    dist.all_reduce(manual)                   # SUM is the default op
    manual /= world_size                      # SUM -> mean: DDP's contract

    rel_err = max_rel_err(manual, snapshot)
    # 1e-2 is a margin, not a rubber stamp: a wrong scale (say you forgot
    # /world_size) is off by ~world_size; a wrong tensor is off by O(1).
    # fp16 summation-order noise between the two paths is ~1e-3.
    claim1 = rel_err < 1e-2
    print(f"[rank {rank}] probe={probe_name} "
          f"grad|max|={snapshot.float().abs().max().item():.3e} rel_err={rel_err:.2e}")

    # ---- claim 2: every rank holds the SAME gradient ----------------
    # (full-tensor all_gather, not a checksum: LoRA-sized probes are
    # tens of KB, so compare bitwise — stronger and still cheap)
    gathered = [torch.empty_like(snapshot) for _ in range(world_size)]
    dist.all_gather(gathered, snapshot)
    claim2 = all(torch.equal(gathered[0], g) for g in gathered[1:])

    # ---- claim 3: every rank steps its OWN optimizer, replicas agree -
    # .grad still holds LOCAL values from claim 1's no_sync pass — stepping
    # now would FORK the replicas (each rank moves its own way). One real
    # synced backward first: that is what a training step actually is.
    policy.zero_grad()
    local_loss(ddp, ids_r, attn_r).backward()
    trainable = [p for p in policy.parameters() if p.requires_grad]  # GRPOTrainer pattern
    opt = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.0)
    # any lr / any rank: the claim is about AGREEMENT, not quality —
    # nobody is in charge, every replica runs its own identical update
    before = probe.detach().clone()
    opt.step()
    moved = not torch.equal(before, probe.detach())  # zero movement = the
    # probe never participated -> the check would be vacuous
    w = probe.detach().clone()
    gathered_w = [torch.empty_like(w) for _ in range(world_size)]
    dist.all_gather(gathered_w, w)
    claim3 = moved and all(torch.equal(gathered_w[0], g) for g in gathered_w[1:])

    return {
        "ddp_eq_manual_avg": claim1,
        "grad_same_across_ranks": claim2,
        "replicas_identical_after_step": claim3,
    }


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

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":  # torchrun re-imports __main__ per worker (P7 lesson)
    main()
