# Phase 3 — Memory Benchmark

Hardware: GTX 1080 Ti 11GB · Model: Qwen2.5-0.5B-Instruct + LoRA r16
(fp32 adapters on frozen fp16 base) · Task: toy arithmetic ·
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

Workload: 6 prompts × group 2, max_new_tokens=384, 5 steps.

Regenerate with:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run -n tinyrl python scripts/benchmark_memory.py --max-steps 5 \
  --n-prompts-per-step 6 --max-new-tokens 384
```

## Results

| variant | peak alloc (GiB) | s/step | notes |
|---|---|---|---|
| baseline | 7.631 | 6.2 | full-batch forward+backward, ref resident on GPU |
| gradient_checkpointing | 7.640 | 5.8 | **no savings** — see Finding 2 |
| ref_cpu_offload | 6.711 | 4.2 | −0.92 GiB ≈ the fp16 ref weights (0.5B × 2 B) |
| micro_batch_2 | 3.990 | 2.8 | −3.64 GiB — the big win |
| everything | 3.990 | 3.1 | identical to micro_batch_2 — see Finding 4 |

s/step is indicative only: after step 1 the variants' fp16 summation order
diverges, so sampled response lengths (and thus step time) drift apart.
The memory numbers are the clean signal.

## Findings

**Finding 1 — the logits tensor, not decoder activations, is the memory
hog.** A forward pass materializes `[N, T, V]` logits (N sequences ×
padded length T × 152k vocab), then `compute_logprobs` makes an fp32 copy
for `log_softmax` — at 12×~180 tokens that is ~0.7 GiB fp16 + ~1.3 GiB
fp32, alive simultaneously. The 0.5B decoder's own activations are small
by comparison; the vocab dimension dominates everything.

**Finding 2 — gradient checkpointing saves nothing under qkvo-LoRA.**
Checkpointing drops activations *inside* decoder blocks, recomputing them
in backward. But with frozen base weights, autograd already saves almost
nothing inside a block: the only tensors it must keep for the LoRA path
are the hidden states at each adapter input, which are block-boundary
values checkpointing keeps anyway. The +0.009 GiB is checkpointing
bookkeeping. The textbook "huge savings" applies to *full* fine-tuning,
where every intermediate activation feeds a parameter gradient. Verified
a second way: the P3.2 correctness run showed bit-identical metrics —
checkpointing works mechanically; it just isn't where this model's memory
is.

**Finding 3 — ref offload's −0.92 GiB is exactly the fp16 reference
weights** (0.5B params × 2 bytes ≈ 0.93 GiB). It helps because the peak
of the non-micro-batched variants happens during the *training* pass,
when the offloaded ref is back on the CPU.

**Finding 4 — micro_batch_2 and everything peak at the *exact same*
3.990 GiB, and that equality is diagnostic.** Micro-batching shrinks the
training pass so much that the peak migrates to `prepare()`: the old/ref
logprob passes still run on the **full batch** (no_grad, transient
logits + fp32 copy), and during those passes the ref model is on the GPU
in *both* variants — so resident memory at the peak moment is identical
and the flags cannot differ. Next lever if we ever need to go below
4 GiB: chunk the logprob passes (or fuse LoRA+lm_head+cross-entropy the
way Liger kernels do) instead of computing the full `[N, T, V]` tensor.

**Methodology note — `max_memory_allocated()` is a process-lifetime
high-water mark.** The first version of this benchmark ran all variants
in one process without `torch.cuda.reset_peak_memory_stats()` between
them; variants 2–5 all reported the *identical* earlier maximum (the
telltale: four equal numbers). Fixed in `scripts/benchmark_memory.py`:
`empty_cache()` + `reset_peak_memory_stats()` before each variant.
