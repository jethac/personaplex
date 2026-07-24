# PersonaPlex on NVIDIA GB10 (DGX Spark / ThinkStation PGX): setup & benchmarking playbook

Validated on: Lenovo ThinkStation PGX — GB10 (sm_121, 48 SMs, 25 MB L2),
aarch64, 128 GB unified LPDDR5X (~273 GB/s), Ubuntu 24.04, driver
595.71.05, CUDA 13.0/13.2. Single machine ("sm_121, n=1") — other GB10
owners: please run the harness below and submit your env.json + summaries.

## 1. Environment
```bash
python3.12 -m venv ~/venv-pp && source ~/venv-pp/bin/activate
pip install -U pip wheel
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
# sphn has no linux-aarch64 wheel: needs rust + cmake
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal && source ~/.cargo/env
CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install -e moshi/.
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: 2.13.0+cu130 True NVIDIA GB10
```

## 2. Triton on sm_121 (needed for --w8a16 / --mimi-fp16 compile paths)
Triton bundles a `ptxas-blackwell` that may mismatch the driver stack;
point it at the system ptxas:
```bash
B=$VIRTUAL_ENV/lib/python3.12/site-packages/triton/backends/nvidia/bin
mv $B/ptxas-blackwell $B/ptxas-blackwell.orig
ln -s /usr/local/cuda/bin/ptxas $B/ptxas-blackwell
```

## 3. Model access
`nvidia/personaplex-7b-v1` is gated: request access on Hugging Face, then
`huggingface-cli login`. Architecture-identical latency proxying with
`kyutai/moshiko-pytorch-bf16` (ungated) matches real-weight latency within
0.1 ms on this machine (see bench/results/ABLATIONS.md notes).

## 4. Benchmarking (headless; never the SSL webserver)
```bash
# protocol run (see bench/protocol.md — pre-registered methodology)
taskset -c 5-9,15-19 python bench/bench.py --frames 500 --out bench/results/my-label
# best-known config
taskset -c 5-9,15-19 python bench/bench.py --fast --frames 500 --out bench/results/my-fast
# 30-min sustained + nvidia-smi dmon capture
taskset -c 5-9,15-19 python bench/bench.py --fast --sustained-minutes 30 --out bench/results/my-soak
# nsys profile of 50 steady-state frames
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
  -t cuda,nvtx,osrt --cuda-graph-trace=node -o trace \
  taskset -c 5-9,15-19 python bench/bench.py --frames 80 --warm-skip 10 --profile-frames 50
# quantization quality gates (bf16 reference vs schemes)
taskset -c 5-9,15-19 python bench/divergence.py --forced --frames 1500 \
  --seeds 42424,1001,2002 --schemes fp8,w8a16 --out bench/results/my-drift
# GEMV bandwidth lab (cold-rotation methodology)
taskset -c 5-9,15-19 python bench/microbench/gemv_lab.py --full --out results-cold.csv
```
Cores 5-9,15-19 are the Cortex-X925 performance cores on the 20-core
Grace CPU; pin to them.

## 5. Known results on this machine (bench/results/ABLATIONS.md)
bf16 stock 101.7 ms/frame -> `--fast` 46.5 ms p50 / 48.1 ms p99.9
(0 misses of the 80 ms budget in 475 frames). Quality gates in
bench/DIVERGENCE.md. GEMV bandwidth study in bench/microbench/README.md.

## 6. Gotchas
- ~135 s cold start is dominated by loading 16.7 GB of safetensors.
- The box shares GPU/CPU with anything else you run: check
  `nvidia-smi` + `ps aux --sort=-%mem` before benchmarking, and record
  co-tenants (bench.py env.json does this automatically).
- torch._scaled_mm on sm_121 uses an sm89 path (126-226 GB/s at decode
  shapes) — prefer --w8a16 over --fp8 on GB10.
