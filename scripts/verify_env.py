"""Phase 0 environment verification.

Checks:
  1. PyTorch import + CUDA availability
  2. GPU device name + capability
  3. Qwen2.5-0.5B-Instruct FP16 inference on GPU

Usage:
    python scripts/verify_env.py
"""

import time
from pathlib import Path

import torch

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-0.5B-Instruct"

PROMPT = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 17 + 26? Answer briefly."},
]


def check_cuda() -> None:
    print("=== CUDA check ===")
    print(f"torch                : {torch.__version__}")
    print(f"cuda available       : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available — check the PyTorch build (need cu118 for GTX 1080 Ti).")
    print(f"device name          : {torch.cuda.get_device_name()}")
    print(f"device capability    : sm_{'.'.join(map(str, torch.cuda.get_device_capability()))}")
    print(f"cuda version (build) : {torch.version.cuda}")
    print(f"total GPU memory     : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")


def check_inference() -> None:
    print("\n=== Qwen2.5-0.5B-Instruct inference check ===")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found at {MODEL_PATH}. Download it first (see README).")

    print(f"loading model from  : {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16).to("cuda")
    model.eval()

    inputs = tokenizer.apply_chat_template(
        PROMPT, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    n_prompt_tokens = inputs["input_ids"].shape[1]
    response = tokenizer.decode(output_ids[0][n_prompt_tokens:], skip_special_tokens=True)
    n_new_tokens = output_ids.shape[1] - n_prompt_tokens
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    print(f"prompt tokens       : {n_prompt_tokens}")
    print(f"generated tokens    : {n_new_tokens}")
    print(f"decode time         : {elapsed:.2f}s  ({n_new_tokens / elapsed:.1f} tokens/s)")
    print(f"peak GPU memory     : {peak_mem:.2f} GiB")
    print(f"response            : {response.strip()!r}")


if __name__ == "__main__":
    check_cuda()
    check_inference()
    print("\nPhase 0 environment verified ✓")
