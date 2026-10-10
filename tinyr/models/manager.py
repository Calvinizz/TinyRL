"""ModelManager: owns tokenizer, LoRA policy, and frozen reference model."""

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinyr.config import MODEL_PATH, TrainingConfig


class ModelManager:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg

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
        # no dropout in Qwen2.5; eval() only affects dropout/batchnorm,
        # gradients still flow through the model in train steps
        policy.eval()

        # TODO P3.2 (part of gradient checkpointing): if
        # cfg.gradient_checkpointing, wrap the policy's forward so that
        # activations are NOT stored — see enable_gradient_checkpointing()
        # below; call it here once implemented.

        # TODO P3.4a: when cfg.ref_model_offload is True the reference
        # model should live on the CPU (it is only needed for a short
        # no-grad pass in GRPOTrainer.prepare, which will move it to the
        # GPU on demand — see TODO P3.4b there). Change the .to(...) below
        # to pick the device accordingly.
        ref = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.float16
        ).to("cpu" if cfg.ref_model_offload else cfg.device).eval()
        for p in ref.parameters():
            p.requires_grad_(False)

        self.tokenizer = tokenizer
        self.policy = policy
        self.ref = ref
        if cfg.gradient_checkpointing:
            self.enable_gradient_checkpointing()
        

    def enable_gradient_checkpointing(self) -> None:
        """TODO P3.2: trade compute for activation memory on the policy.

        Two things are required (both one-liners on a PEFT model):
        1. self.policy.gradient_checkpointing_enable()  — recompute
           activations during backward instead of storing them
        2. self.policy.enable_input_require_grads()     — WITHOUT this,
           gradients do not flow back through the checkpointed segments
           for a PEFT model (the frozen embedding output has
           requires_grad=False, and checkpointing only saves tensors
           that need grad; the hook makes the embedding OUTPUT require
           grad so the checkpointed blocks get recomputed)

        Expected effect: activation memory drops sharply, step time rises
        ~20-40%. Verify with scripts/benchmark_memory.py.
        """
        self.policy.gradient_checkpointing_enable()
        self.policy.enable_input_require_grads()
