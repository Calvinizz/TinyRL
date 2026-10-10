"""Phase 1 — Minimal GRPO trainer (single file, education-first).

One complete GRPO optimization step per iteration:

    sample prompts
    -> generate `group_size` responses per prompt          (rollout)
    -> rule-based rewards (GSM8K exact answer)             [TODO 2]
    -> group-relative advantages                           [TODO 3]
    -> old / reference / new log probs on response tokens  [TODO 4, 5]
    -> clipped policy loss + KL penalty                    [TODO 6]
    -> optimizer step

The framework (data, models, rollout loop, optimizer, logging) is complete.
The GRPO core is marked TODO 1-6 — fill them in using the docstrings.

Run:
    python scripts/train_grpo.py --toy --max-steps 30 --lr 1e-4   # toy task first
    python scripts/train_grpo.py --max-steps 50                   # GSM8K subset
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
DATA_PATH = ROOT / "data" / "gsm8k_prompts.jsonl"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Solve the math problem step by step. "
    "You MUST end your response with '#### <final answer>' where the final "
    "answer is a single number, for example:\n"
    "24 + 11 = 35\n"
    "#### 35"
)


@dataclass
class GRPOConfig:
    # rollout
    n_prompts_per_step: int = 4
    group_size: int = 2  # responses sampled per prompt (README: 2~4)
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    # LoRA (README recommended first experiment; full-FT fp16 + AdamW
    # does not fit / numerically work on 11GB — revisit in Phase 3)
    lora_rank: int = 16
    lora_alpha: int = 32
    # optimization (LoRA lr is much larger than full-FT GRPO lr)
    lr: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    clip_eps: float = 0.2
    kl_coef: float = 0.04
    adv_eps: float = 1e-4
    n_inner_epochs: int = 2  # >1 makes the importance ratio non-trivial
    # run
    max_steps: int = 50
    seed: int = 0
    device: str = "cuda"


# =====================================================================
# TODO 1-6: the GRPO core. Implement these.
# =====================================================================


def extract_answer(response: str) -> str | None:
    """TODO 1: parse the model's final answer from a response string.

    The prompt asks the model to end with '#### <number>'.
    Return the answer string (e.g. '42'), or None if no answer is found.
    A reasonable fallback: the last number appearing in the text.
    """
    # TODO(you): raise NotImplementedError removed once implemented
    match = re.search(r'####\s*(\d+)',response)
    if match:
        return match.group(1)
    numbers = re.findall(r'-?\d+', response)   # fallback: last number
    return numbers[-1] if numbers else None


def compute_rewards(
    responses: list[str], ground_truths: list[str]
) -> torch.Tensor:
    """TODO 2: rule-based reward — GSM8K exact-answer match.

    Args:
        responses:      list of N response strings (N = n_prompts * group_size)
        ground_truths:  list of N ground-truth answer strings
                        (each prompt's answer repeated group_size times)
    Returns:
        torch.Tensor[float] of shape [N]: 1.0 for exact match, else 0.0
    """
    rewards = torch.zeros(len(responses), dtype=torch.float32)
    for i in range(len(responses)):
        resp = extract_answer(responses[i])
        gt = ground_truths[i]
        rewards[i] = 1.0 if resp is not None and resp == gt else 0.0
    return rewards
    



def compute_group_advantages(
    rewards: torch.Tensor, group_size: int, eps: float = 1e-4
) -> torch.Tensor:
    """TODO 3: group-relative advantage.

        A_i = (r_i - mean(r_group)) / (std(r_group) + eps)

    Args:
        rewards: Tensor[float] of shape [N], laid out as consecutive groups:
                 [prompt_0 x G, prompt_1 x G, ...]  (N = n_prompts * G)
    Returns:
        Tensor[float] of shape [N]
    Hints:
        reshape to [n_prompts, group_size], use dim=1 statistics
        (std with unbiased=False), flatten back.
    """
    advantage = torch.zeros_like(rewards)
    rewards = rewards.reshape(len(rewards)//group_size,group_size)
    for i in range(len(rewards)):
        mean = rewards[i].mean()
        std = rewards[i].std(unbiased=False)
        advantage[i*group_size : (i+1)*group_size] = (rewards[i] - mean)/(std+eps)
    return advantage

def build_response_mask(
    attention_mask: torch.Tensor, prompt_width: int
) -> torch.Tensor:
    """TODO 4: mask out everything except response tokens.

    Sequences are laid out as: [left-pad][prompt][response].
    `attention_mask` is already 0 on left padding and on tokens generated
    after the first EOS. Response tokens start at column `prompt_width`.

    Args:
        attention_mask: Tensor[bool] or [long] of shape [N, T]
        prompt_width:   number of columns the (padded) prompt occupies
    Returns:
        Tensor[bool] of shape [N, T]: True on response tokens only
    Hints:
        torch.arange(T) >= prompt_width, combined with attention_mask.bool()
    """
    N,T = attention_mask.shape
    position = torch.arange(T,device=attention_mask.device)
    resp_position = position >= prompt_width
    resp_position = resp_position & attention_mask.bool()
    return resp_position




def compute_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """TODO 5: per-token log probabilities of the response tokens.

    For each position t, the model predicts token t+1 from tokens <= t.
    So logits[:, :-1] predict input_ids[:, 1:].

    Args:
        model:          a causal LM (policy, old policy = same weights at
                        rollout time, or reference model)
        input_ids:      LongTensor [N, T]
        attention_mask: LongTensor [N, T]
        response_mask:  BoolTensor  [N, T] (True on response positions)
    Returns:
        Tensor[float] [N, T-1]: column j is logp(input_ids[:, j+1] given
        prefix <= j), zero at non-response positions (shifted convention:
        mask with response_mask[:, 1:])
    Hints:
        - logits = model(...).logits[:, :-1]  -> cast .float() BEFORE
          log_softmax (fp16 softmax loses precision; the [N, T, V] fp32
          tensor is the memory bottleneck — keep the batch small)
        - per-token logp = log_softmax(logits, dim=-1).gather(
              -1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        - align the mask: token at column t is "chosen" at shifted
          position t-1, so use response_mask[:, 1:]
        - call sites: old/ref logprobs run under torch.no_grad();
          new logprobs need gradients — do NOT add no_grad here.
    """
    logits = model(input_ids = input_ids,attention_mask = attention_mask).logits[:, :-1]
    label = input_ids[:,1:]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_logps = log_probs.gather(
        dim=-1,
        index=label.unsqueeze(-1),
    ).squeeze(-1)
    # gather结束变成  N T 了
    token_logps = token_logps * response_mask[:,1:]
    return token_logps

def compute_grpo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    cfg: GRPOConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """TODO 6: the GRPO policy loss (per-token means, masked to responses).

        ratio    = exp(logp_new - logp_old)
        surrogate = min(ratio * A, clamp(ratio, 1-eps, 1+eps) * A)
        KL (k3)  = exp(logp_ref - logp_new) - (logp_ref - logp_new) - 1
        loss     = -mean_over_response_tokens(surrogate - kl_coef * KL)

    Args: logprobs have shape [N, T-1] (the shifted convention returned
          by compute_logprobs); advantages [N]; response_mask [N, T] —
          align it with response_mask[:, 1:] before masking, and divide
          by the response-token count, not by N*T)
    Returns:
        (loss_scalar, metrics) with keys:
        "ratio_max", "ratio_min", "clip_frac", "kl"
    Hints:
        - masked mean: (x * mask).sum() / mask.sum()
        - clip_frac: fraction of response tokens where the ratio leaves
          the [1-eps, 1+eps] band
    """
    mask  = response_mask[:,1:].float()
    ratio = torch.exp(new_logprobs - old_logprobs)
    # 应用到所有 token
    adv = advantages.unsqueeze(-1)
    surr = ratio * adv
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - cfg.clip_eps,
        1.0 + cfg.clip_eps,
    )
    surr1 = clipped_ratio * adv
    surrogate = torch.minimum(surr,surr1)
    log_ratio_ref = ref_logprobs - new_logprobs
    kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1
    
    objective = surrogate - cfg.kl_coef * kl
    loss_tokens = -objective

    denom = mask.sum().clamp_min(1.0)

    loss = (loss_tokens * mask).sum() / denom
    metrics = {
        "ratio_max": float(ratio[mask.bool()].max().item()),
        "ratio_min": float(ratio[mask.bool()].min().item()),
        "clip_frac": float((((ratio - 1).abs() > cfg.clip_eps).float() * mask).sum().item() / denom.item()),
        "kl":        float((kl * mask).sum().item() / denom.item()),
    }
    # resp token求平均loss
    return loss, metrics


# =====================================================================
# Framework below — data, models, rollout loop, training loop, logging.
# =====================================================================


def load_prompts(limit: int | None = None) -> list[dict]:
    if not DATA_PATH.exists():
        raise SystemExit(
            f"{DATA_PATH} not found — run `python scripts/prepare_gsm8k.py` first"
        )
    records = [json.loads(line) for line in DATA_PATH.open()]
    return records[:limit] if limit else records


def make_toy_prompts(n: int = 16) -> list[dict]:
    """Small deterministic arithmetic task with a dense learnable signal."""
    rng = random.Random(1234)
    prompts = []
    for _ in range(n):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        prompts.append(
            {"question": f"What is {a} + {b}?", "answer": str(a + b)}
        )
    return prompts


def load_policy_and_ref(cfg: GRPOConfig):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched generation

    policy = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16
    ).to(cfg.device)
    policy = get_peft_model(
        policy,
        LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    # keep the trainable adapters in fp32: updates of size lr~1e-5 would
    # round to zero in fp16 (the frozen fp16 base stays as is)
    for p in policy.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    policy.print_trainable_parameters()
    # Qwen2.5 has no dropout; eval() keeps rollout deterministic in behavior
    # (gradients still flow — eval() only affects dropout/batchnorm).
    policy.eval()

    ref = (
        AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16)
        .to(cfg.device)
        .eval()
    )
    for p in ref.parameters():
        p.requires_grad_(False)

    return policy, ref, tokenizer


@torch.no_grad()
def rollout(
    policy,
    tokenizer,
    questions: list[str],
    cfg: GRPOConfig,
) -> dict:
    """Sample `group_size` responses per prompt. Returns padded batch.

    Layout per row: [left-pad][prompt][response]; rows are grouped as
    [q0]*G, [q1]*G, ... which matches compute_group_advantages' reshape.
    """
    G, N = cfg.group_size, len(questions) * cfg.group_size

    encoded = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
            add_generation_prompt=True,
            tokenize=True,
        )["input_ids"]
        for q in questions
    ]
    batch = tokenizer.pad({"input_ids": encoded}, padding=True, return_tensors="pt")
    input_ids = batch["input_ids"].repeat_interleave(G, dim=0).to(cfg.device)
    attention_mask = (
        batch["attention_mask"].repeat_interleave(G, dim=0).to(cfg.device)
    )
    prompt_width = input_ids.shape[1]

    output_ids = policy.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=True,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=0,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = output_ids[:, prompt_width:]

    # attention on generated tokens: 1 up to and including first EOS, then 0
    gen_attention = torch.ones_like(generated)
    for i in range(generated.shape[0]):
        eos_pos = (generated[i] == tokenizer.eos_token_id).nonzero()
        if len(eos_pos) > 0:
            gen_attention[i, eos_pos[0, 0] + 1 :] = 0

    input_ids = torch.cat([input_ids, generated], dim=1)
    attention_mask = torch.cat([attention_mask, gen_attention], dim=1)
    responses = tokenizer.batch_decode(generated, skip_special_tokens=True)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_width": prompt_width,
        "responses": responses,
    }


def verify_gradients(policy) -> None:
    """Checklist: 'Verify gradients are non-zero'."""
    n_nonzero = 0
    n_total = 0
    for p in policy.parameters():
        if p.requires_grad and p.grad is not None:
            n_total += 1
            if p.grad.abs().sum() > 0:
                n_nonzero += 1
    print(f"  grads non-zero: {n_nonzero}/{n_total} parameters")
    if n_nonzero == 0:
        print("  WARNING: all gradients are zero — check masking / loss sign")


def train(cfg: GRPOConfig, prompts: list[dict]) -> None:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    policy, ref, tokenizer = load_policy_and_ref(cfg)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    # no GradScaler: LoRA grads live in fp32 on small params, and the
    # scaler refuses to unscale fp16 (full-model) gradients anyway

    print(
        f"training on {len(prompts)} prompts | "
        f"batch={cfg.n_prompts_per_step}x{cfg.group_size} | lr={cfg.lr}"
    )

    for step in range(1, cfg.max_steps + 1):
        t0 = time.perf_counter()
        questions = random.sample(prompts, cfg.n_prompts_per_step)
        ground_truths = [r["answer"] for r in questions for _ in range(cfg.group_size)]

        # ---- rollout ----
        roll = rollout(policy, tokenizer, [r["question"] for r in questions], cfg)
        responses = roll["responses"]

        # ---- reward & advantage (TODO 2, 3) ----
        rewards = compute_rewards(responses, ground_truths).to(cfg.device)
        advantages = compute_group_advantages(
            rewards, cfg.group_size, eps=cfg.adv_eps
        ).to(cfg.device)

        # ---- masks & logprobs (TODO 4, 5) ----
        response_mask = build_response_mask(
            roll["attention_mask"], roll["prompt_width"]
        )
        # old/ref logprobs are constants w.r.t. the update — compute them
        # without an autograd graph. A stray graph keeps the fp32
        # [N, T, V] log_softmax tensor alive until backward and OOMs an
        # 11GB GPU (only the training forward below needs gradients).
        with torch.no_grad():
            old_logprobs = compute_logprobs(
                policy, roll["input_ids"], roll["attention_mask"], response_mask
            )
            ref_logprobs = compute_logprobs(
                ref, roll["input_ids"], roll["attention_mask"], response_mask
            )

        # ---- policy updates (TODO 6) ----
        for inner_epoch in range(cfg.n_inner_epochs):
            new_logprobs = compute_logprobs(
                policy, roll["input_ids"], roll["attention_mask"], response_mask
            )
            loss, metrics = compute_grpo_loss(
                new_logprobs, old_logprobs, ref_logprobs,
                advantages, response_mask, cfg,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if step == 1 and inner_epoch == 0:
                verify_gradients(policy)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, cfg.max_grad_norm
            )
            optimizer.step()

        # ---- logging ----
        n_resp_tokens = response_mask.sum().item()
        acc = (rewards > 0).float().mean().item()
        print(
            f"step {step:3d} | reward {rewards.mean().item():.3f} "
            f"(acc {acc:.2f}) | loss {loss.item():+.4f} "
            f"| kl {metrics.get('kl', float('nan')):.4f} "
            f"| clip {metrics.get('clip_frac', float('nan')):.3f} "
            f"| ratio [{metrics.get('ratio_min', float('nan')):.2f}, "
            f"{metrics.get('ratio_max', float('nan')):.2f}] "
            f"| grad {grad_norm:.2f} | resp_tok {n_resp_tokens} "
            f"| {time.perf_counter() - t0:.1f}s"
        )
        print(f"  sample response: {responses[0][:120]!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    for field, default in GRPOConfig.__dict__.items():
        if not field.startswith("_"):
            parser.add_argument(f"--{field.replace('_', '-')}", type=type(default), default=default)
    parser.add_argument("--toy", action="store_true", help="use the toy arithmetic task")
    parser.add_argument("--n-train", type=int, default=None, help="limit number of prompts")
    args = parser.parse_args()

    cfg = GRPOConfig(**{
        k: v for k, v in vars(args).items() if k in GRPOConfig.__dict__
    })
    prompts = make_toy_prompts() if args.toy else load_prompts(args.n_train)
    train(cfg, prompts)


if __name__ == "__main__":
    main()
