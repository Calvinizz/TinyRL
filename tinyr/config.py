"""Shared configuration and paths for TinyRL."""

from dataclasses import dataclass
from pathlib import Path

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
class TrainingConfig:
    # rollout
    n_prompts_per_step: int = 4
    group_size: int = 2  # responses sampled per prompt (README: 2~4)
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    # LoRA (README recommended first experiment)
    lora_rank: int = 16
    lora_alpha: int = 32
    # optimization
    lr: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    clip_eps: float = 0.2
    kl_coef: float = 0.04
    adv_eps: float = 1e-4
    n_inner_epochs: int = 2
    # system
    buffer_max_size: int = 8
    max_steps: int = 50
    seed: int = 0
    device: str = "cuda"
