# TinyRL @ 390 server — source before any GPU work:  . scripts/server_env.sh
export VLLM_USE_FLASHINFER_SAMPLER=0   # flashinfer JIT needs nvcc>=12, system nvcc is 10.1 (2019)
export XDG_CACHE_HOME=/data/zy/cache   # /home is 99% full; keep torch.compile/vllm caches on /data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn  # fork + threaded parent (transformers/tokenizers) = EngineCore futex deadlock
