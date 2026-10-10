"""GRPOTrainer: consumes Experiences and updates the LoRA policy."""

import torch

from tinyr.config import TrainingConfig
from tinyr.experience.experience import Experience, iter_micro_batches


class GRPOTrainer:
    def __init__(self, policy, ref_model, cfg: TrainingConfig):
        self.policy = policy
        self.ref = ref_model
        self.cfg = cfg
        self.trainable = [p for p in policy.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.trainable, lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self._steps = 0

    # ------------------------------------------------------------------
    # TODO 5-7: the GRPO core — port from scripts/train_grpo.py (Phase 1).
    # ------------------------------------------------------------------

    @staticmethod
    def compute_group_advantages(
        rewards: torch.Tensor, group_size: int, eps: float = 1e-4
    ) -> torch.Tensor:
        """TODO 5: A_i = (r_i - mean(r_group)) / (std(r_group) + eps).

        Args: rewards FloatTensor [N] laid out as consecutive groups
              [prompt_0 x G, prompt_1 x G, ...]
        Returns: FloatTensor [N]
        (staticmethod so unit tests can call it without models loaded)
        """
        rewards = rewards.reshape(len(rewards) // group_size,group_size)
        adv = torch.zeros_like(rewards)
        for i in range(len(adv)):
            adv[i] = (rewards[i] - rewards[i].mean()) / (rewards[i].std(unbiased=False) + eps)
        adv = adv.flatten()
        return adv


    def compute_logprobs(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """TODO 6: per-token response log probabilities.

        Returns FloatTensor [N, T-1], column j = logp(input_ids[:, j+1] |
        prefix <= j), zero at non-response positions. Cast logits .float()
        BEFORE log_softmax; mask with response_mask[:, 1:]; never wrap in
        no_grad here — call sites decide (prepare() wraps old/ref).
        """
        logits = model(input_ids=input_ids,attention_mask=attention_mask).logits[:,:-1]
        label = input_ids[:,1:]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        token_logps = log_probs.gather(
            dim = -1,
            index = label.unsqueeze(-1),
        ).squeeze(-1)
        token_logps = token_logps * response_mask[:,1:]
        return token_logps
    
    def compute_grpo_loss(
        self,
        new_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        cfg: TrainingConfig,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """TODO 7: ratio + PPO clipping + k3 KL penalty, masked to responses.

            ratio     = exp(logp_new - logp_old)
            surrogate = min(ratio * A, clamp(ratio, 1-eps, 1+eps) * A)
            KL (k3)   = exp(logp_ref - logp_new) - (logp_ref - logp_new) - 1
            loss      = -mean_over_response_tokens(surrogate - kl_coef * KL)

        logprobs are [N, T-1] (shifted); align mask with response_mask[:, 1:].
        Returns (loss_scalar, metrics) with float keys:
        "ratio_max", "ratio_min", "clip_frac", "kl".
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
        return loss,metrics
    # ------------------------------------------------------------------
    # Framework: experience preparation and the update loop.
    # ------------------------------------------------------------------

    def prepare(self, exp: Experience) -> Experience:
        """Fill advantages and old/ref logprobs (constants for the update).

        TODO P3.4b: when the reference model is CPU-offloaded
        (cfg.ref_model_offload), self.ref lives on the CPU. Move it to
        cfg.device for the no-grad ref pass below, then move it back —
        use try/finally so an exception mid-pass cannot strand ~1GB of
        fp16 weights on the GPU. Wrapping only the ref pass keeps the
        policy pass untouched.
        """
        exp.advantages = self.compute_group_advantages(
            exp.rewards, self.cfg.group_size, eps=self.cfg.adv_eps
        )
        # 开启cpu offload , 临时从cpu搬回gpu
        if self.cfg.ref_model_offload:
            self.ref.to(self.cfg.device)
        try:
            with torch.no_grad():
                exp.old_logprobs = self.compute_logprobs(
                    self.policy, exp.input_ids, exp.attention_mask, exp.response_mask
                )
                exp.ref_logprobs = self.compute_logprobs(
                    self.ref, exp.input_ids, exp.attention_mask, exp.response_mask
                )
        finally:
            if self.cfg.ref_model_offload:
                self.ref.to("cpu")
        return exp

    def train_step(self, exp: Experience) -> dict[str, float]:
        """Run n_inner_epochs policy updates on one experience.

        TODO P3.3: micro-batching + gradient accumulation.

        Structure (the framework part is already wired):

            for inner_epoch in range(n_inner_epochs):
                zero_grad
                for rows in iter_micro_batches(N, cfg.micro_batch_size):
                    sub  = exp.rows(rows)
                    forward -> loss on the micro-batch
                    (loss / n_micro_batches).backward()     # ← the TODO
                clip + optimizer.step()                     # ONCE per epoch

        The contract your code must satisfy (tests/test_accumulation.py):
        with equal response-token counts per micro-batch, accumulated
        micro-batch gradients must EQUAL full-batch gradients. That is
        why each micro-batch's loss is divided by the NUMBER OF
        MICRO-BATCHES — grads then sum to the full-batch mean, not to
        n_micro copies of it. Keep optimizer.step() OUT of the micro
        loop; zero_grad ONCE per inner epoch (not per micro-batch).

        Metrics: keep returning the last micro-batch's dict + grad_norm.
        """
        metrics: dict[str, float] = {}
        for inner_epoch in range(self.cfg.n_inner_epochs):
            micro_batches = list(iter_micro_batches(
                exp.n_sequences, self.cfg.micro_batch_size
            ))
            self.optimizer.zero_grad(set_to_none=True)
            loss_all = 0
            for rows in micro_batches:
                sub = exp.rows(rows)
                new_logprobs = self.compute_logprobs(
                    self.policy, sub.input_ids, sub.attention_mask,
                    sub.response_mask,
                )
                loss, metrics = self.compute_grpo_loss(
                    new_logprobs, sub.old_logprobs, sub.ref_logprobs,
                    sub.advantages, sub.response_mask, self.cfg,
                )
                # TODO P3.3: backward with the accumulation-correct scale
                loss = loss / len(micro_batches)
                loss.backward()

            if self._steps == 0 and inner_epoch == 0:
                self.verify_gradients()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.trainable, self.cfg.max_grad_norm
            )
            self.optimizer.step()
            self._steps += 1
            metrics["grad_norm"] = float(grad_norm.item())
        return metrics

    def verify_gradients(self) -> None:
        n_nonzero = sum(
            1
            for p in self.trainable
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        print(f"  grads non-zero: {n_nonzero}/{len(self.trainable)} parameters")
        if n_nonzero == 0:
            print("  WARNING: all gradients are zero — check masking / reward")
