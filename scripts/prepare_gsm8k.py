"""Download a small GSM8K subset and save prompts for Phase 1 training.

Output: data/gsm8k_prompts.jsonl  (one {"question", "answer"} per line)

Run:
    python scripts/prepare_gsm8k.py [--n 500]
"""

import argparse
import json
import os
from pathlib import Path

# Must be set before `datasets` / `huggingface_hub` is imported — this machine
# cannot reach huggingface.co directly. Override by exporting HF_ENDPOINT.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "gsm8k_prompts.jsonl"


def extract_ground_truth(answer_field: str) -> str:
    """GSM8K answers look like '<reasoning>... #### 42' -> '42'."""
    final = answer_field.split("####")[-1].strip().replace(",", "")
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="number of prompts to keep")
    args = parser.parse_args()

    ds = load_dataset("openai/gsm8k", "main", split="train")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with OUT_PATH.open("w") as f:
        for row in ds:
            if n_written >= args.n:
                break
            record = {
                "question": row["question"].strip(),
                "answer": extract_ground_truth(row["answer"]),
            }
            f.write(json.dumps(record) + "\n")
            n_written += 1

    print(f"wrote {n_written} prompts -> {OUT_PATH}")
    print(f"example: {json.loads(OUT_PATH.open().readline())}")


if __name__ == "__main__":
    main()
