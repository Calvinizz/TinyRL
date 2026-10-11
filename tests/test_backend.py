"""Spec tests for Phase 7: the RolloutBackend contract.

The HF-backend tests run against STUB policy/tokenizer objects that
mimic the transformers surface the backend touches (apply_chat_template
/ pad / batch_decode / generate) — no model, no GPU, red until the
P7.1 port produces the contract tensors. The vLLM tests self-skip
without vllm+cuda and turn red on the 390 server until P7.2 lands.
"""

import pytest
import torch

from tinyr.config import TrainingConfig
from tinyr.rollout import (
    HFRolloutBackend,
    RolloutBackend,
    RolloutResult,
    RolloutWorker,
    VLLMRolloutBackend,
)


def make_result() -> RolloutResult:
    return RolloutResult(
        prompt_ids=torch.zeros(1, 2, dtype=torch.long),
        response_ids=torch.zeros(1, 3, dtype=torch.long),
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        responses=["x"],
        prompt_width=2,
    )


# ---------------------------------------------------------------- contract


class TestContract:
    def test_interface_is_abstract(self):
        with pytest.raises(TypeError):
            RolloutBackend()  # type: ignore[abstract]

    def test_backends_subclass_the_interface(self):
        assert issubclass(HFRolloutBackend, RolloutBackend)
        assert issubclass(VLLMRolloutBackend, RolloutBackend)


class TestWorkerDelegates:
    def test_explicit_backend_wins(self):
        seen = []

        class Recording(RolloutBackend):
            def generate(self, questions):
                seen.append(list(questions))
                return make_result()

        cfg = TrainingConfig(device="cpu")
        worker = RolloutWorker(None, None, cfg, backend=Recording())
        out = worker.generate(["q1", "q2"])
        assert isinstance(out, RolloutResult)
        assert seen == [["q1", "q2"]]

    def test_default_backend_is_hf(self):
        cfg = TrainingConfig(device="cpu")
        worker = RolloutWorker(object(), object(), cfg)
        assert isinstance(worker.backend, HFRolloutBackend)


# ---------------------------------------------------------------- P7.1 (HF)
# Stubs mirror EXACTLY the transformers surface the backend may touch —
# anything else (calling .to() on tokenizer, right-padding, etc.) fails
# loudly with AttributeError here.


class StubTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        """Question length decides prompt length: 'A?' -> 2 ids, 'BB?' -> 3."""
        text = messages[-1]["content"]
        length = 2 + len(text) % 2
        return {"input_ids": [10 + i for i in range(length)]}

    def pad(self, features, padding=True, return_tensors="pt"):
        ids = features["input_ids"]
        W = max(len(x) for x in ids)
        # LEFT padding — ModelManager configures the real tokenizer the
        # same way; generate() needs prompts at the right edge
        input_ids = torch.tensor(
            [[self.pad_token_id] * (W - len(x)) + x for x in ids],
            dtype=torch.long,
        )
        attention = torch.tensor(
            [[0] * (W - len(x)) + [1] * len(x) for x in ids], dtype=torch.long
        )
        return {"input_ids": input_ids, "attention_mask": attention}

    def batch_decode(self, ids, skip_special_tokens=True):
        return [" ".join(str(int(t)) for t in row) for row in ids]


class StubPolicy:
    """Echoes the prompt, then a fixed response: [t0, t1, EOS, 99, 99]."""

    def generate(
        self, input_ids=None, attention_mask=None, max_new_tokens=None, **kw
    ):
        N, W = input_ids.shape
        resp = [30, 31, StubTokenizer.eos_token_id, 99, 99][:max_new_tokens]
        rows = [input_ids[i].tolist() + resp for i in range(N)]
        return torch.tensor(rows, dtype=torch.long)


@pytest.fixture
def hf_backend():
    cfg = TrainingConfig(
        group_size=2, max_new_tokens=5, device="cpu",
        temperature=1.0, top_p=1.0,
    )
    return HFRolloutBackend(StubPolicy(), StubTokenizer(), cfg)


class TestHFBackend:
    def test_shapes_and_group_order(self, hf_backend):
        # 'A?' -> prompt len 2, 'BB?' -> len 3 => W=3, N=2*2=4
        out = hf_backend.generate(["A?", "BB?"])
        assert out.prompt_ids.shape == (4, 3)
        assert out.response_ids.shape == (4, 5)
        assert out.attention_mask.shape == (4, 3 + 5)
        assert out.prompt_width == 3
        # question-major: rows 0,1 are q0 (same prompt), rows 2,3 are q1
        assert torch.equal(out.prompt_ids[0], out.prompt_ids[1])
        assert torch.equal(out.prompt_ids[2], out.prompt_ids[3])
        assert not torch.equal(out.prompt_ids[0], out.prompt_ids[2])

    def test_left_padding(self, hf_backend):
        out = hf_backend.generate(["A?", "BB?"])
        # q0's prompt (2 ids) is left-padded inside W=3: col 0 is pad+0
        assert out.prompt_ids[0, 0].item() == StubTokenizer.pad_token_id
        assert out.attention_mask[0, 0].item() == 0
        assert out.attention_mask[0, 1].item() == 1

    def test_attention_zero_after_first_eos(self, hf_backend):
        out = hf_backend.generate(["A?"])
        # response cols (3..7): [30, 31, EOS, 99, 99] -> 1,1,1,0,0
        resp_att = out.attention_mask[0, out.prompt_width:]
        assert resp_att.tolist() == [1, 1, 1, 0, 0]

    def test_responses_decoded(self, hf_backend):
        out = hf_backend.generate(["A?"])  # 1 question x G=2 -> 2 rows
        assert len(out.responses) == 2
        assert all(isinstance(s, str) for s in out.responses)


# ---------------------------------------------------------------- P7.2 (vLLM)
# Self-skipping on the 1080 Ti; on the 390 server these are the red
# TODO markers until P7.2 is implemented.


class TestVLLMBackend:
    def test_generate_contract(self):
        pytest.importorskip("vllm")
        if not torch.cuda.is_available():
            pytest.skip("needs cuda")
        from tinyr.config import MODEL_PATH
        from transformers import AutoTokenizer

        cfg = TrainingConfig(
            group_size=2, max_new_tokens=16, device="cuda", seed=0,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        backend = VLLMRolloutBackend(str(MODEL_PATH), tokenizer, cfg)
        out = backend.generate(["What is 2 + 3?", "What is 4 + 5?"])
        N, W = out.prompt_ids.shape
        assert N == 4 and out.prompt_width == W
        assert out.response_ids.shape == (N, out.response_ids.shape[1])
        assert out.attention_mask.shape == (
            N, W + out.response_ids.shape[1]
        )
        # per-row attention must match 0* 1* 0* (left pad, real tokens,
        # post-EOS pad) — once it falls back to 0 it may never rise:
        # build_response_mask and rows-trim silently rely on this
        for row in out.attention_mask.tolist():
            seen_one, closed = False, False
            for a in row:
                if a == 1:
                    assert not closed, "0*1*0* violated"
                    seen_one = True
                elif seen_one:
                    closed = True


# ---------------------------------------------------------------- P7.3 (sync)


class TestSyncContract:
    def test_default_sync_is_free_for_shared_weight_backends(self):
        class Recording(RolloutBackend):
            def generate(self, questions):
                return make_result()

        backend = Recording()
        assert backend.served_version is None
        # the no-op must not even look at the path (HF shares the
        # policy object — there is nothing to load)
        backend.sync_policy(7, "/nonexistent/path")
        assert backend.served_version is None


class TestVLLMWeightSync:
    """P7.3 acceptance: weights must FLOW and be applied exactly.

    Two probes, one engine boot:
      * zero-adapter identity — a fresh peft LoRA has lora_B = 0, so
        base + B@A == base mathematically; greedy outputs must be
        token-identical to the unsynced engine. This catches scale
        (alpha/r misread), dtype, and module-mapping bugs — any of
        them perturbs logits and greedy flips somewhere.
      * big perturbation — filling lora_B with noise must change
        outputs; if it doesn't, sync is a silent no-op (the classic
        unversioned-LoRARequest-name cache trap).
    """

    def test_sync_flow(self, tmp_path):
        pytest.importorskip("vllm")
        if not torch.cuda.is_available():
            pytest.skip("needs cuda")
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from tinyr.config import MODEL_PATH

        cfg = TrainingConfig(  # greedy: sampling noise off, weights only
            # vLLM forbids n>1 with temperature=0 (identical greedy
            # samples are pointless) — so G=1 for this probe; N = 2 rows
            group_size=1, max_new_tokens=16, device="cuda", seed=0,
            temperature=0.0,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        backend = VLLMRolloutBackend(str(MODEL_PATH), tokenizer, cfg)
        questions = ["What is 2 + 3?", "What is 4 + 5?"]

        base = backend.generate(questions)  # engine's static base weights

        policy = get_peft_model(
            AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, dtype=torch.float16
            ),
            LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
            ),
        ).eval()

        d1 = tmp_path / "v1"
        policy.save_pretrained(d1)
        backend.sync_policy(1, str(d1))
        assert backend.served_version == 1
        zero = backend.generate(questions)
        assert torch.equal(zero.response_ids, base.response_ids), (
            "zero adapter (lora_B == 0) changed greedy outputs — the "
            "engine is not applying the adapter identity; suspect "
            "scale/dtype/module mapping"
        )

        with torch.no_grad():
            for name, p in policy.named_parameters():
                if "lora_B" in name:
                    p.normal_(0.0, 0.5)
        d2 = tmp_path / "v2"
        policy.save_pretrained(d2)
        backend.sync_policy(2, str(d2))
        assert backend.served_version == 2
        perturbed = backend.generate(questions)
        assert not torch.equal(perturbed.response_ids, base.response_ids), (
            "big lora_B perturbation changed nothing — sync is a silent "
            "no-op; suspect the unversioned adapter-name cache"
        )
