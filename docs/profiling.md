# Profiling

CUDA extensions are compiled with `-lineinfo` and profiler-friendly names. Use
`nsight-python` from a CUDA/Linux machine to collect a reproducible profile from
Python rather than wrapping a benchmark with an ad-hoc `ncu` command.

## Prerequisites

- A CUDA-capable NVIDIA GPU on Linux.
- NVIDIA Nsight Compute (`ncu`) installed and available on `PATH`.
- A CUDA-compatible PyTorch installation.

`nsight-python` is declared as a Linux-only project dependency, so macOS users
can keep editing the lab without trying to install or run the profiler. The
Python package drives Nsight Compute; it does not bundle Nsight Compute itself.

## Run a profile

Each profiled kernel folder provides a `profile.py` script. For example:

```bash
uv run cuda-lab profile 01_vecadd
uv run cuda-lab profile 02_matmul
```

On hosts that restrict performance counters, use the helper from
`scripts/setup_cuda_env.sh`:

```bash
source scripts/setup_cuda_env.sh
cuda_lab_profile 01_vecadd
cuda_lab_profile 02_matmul
```

## Config-driven line charts

Profile scripts follow the nsight-python pattern:

1. **`configs=`** — a list of input sizes (or argument tuples) to sweep.
2. **`nsight.annotate(...)`** — one labeled region per kernel launch; each label
   becomes a series on the chart.
3. **`@nsight.analyze.plot(...)`** — automatic line chart (default) saved under
   `results/profiles/<kernel>/`.

Example shape (see `01_vecadd/profile.py` and `02_matmul/profile.py`):

```python
SIZES = [(n,) for n in range(256, 2049, 256)]

@nsight.analyze.plot(
    filename=str(PROFILE_IMAGE),
    title="Matmul Execution Time",
    plot_type="line",
)
@nsight.analyze.kernel(configs=SIZES, runs=5, ...)
def profile_matmul(n: int) -> None:
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")
    with nsight.annotate("torch.matmul"):
        matmul_ref(a, b)
    with nsight.annotate("tiled"):
        module.matmul_tiled(a, b)
```

Outputs land under `results/profiles/<kernel>/`: aggregated CSV, PNG line chart,
Nsight Compute report, and log.

Add a `profile.py` alongside a new kernel's `bench.py`. Annotate exactly one
kernel launch per region when possible. If an annotation launches several
kernels (common for `torch.matmul` / cuBLAS), set `replay_mode="range"` and
`combine_kernel_metrics=lambda x, y: x + y` so durations sum cleanly. Also avoid
allocating with `torch.zeros` inside an annotated region — that launches an
extra fill kernel; prefer `torch.empty` when the op fully overwrites the output.

Use `ncu --query-metrics` on the CUDA machine only when you need to discover
additional hardware metrics for a profile script. The standard `profile` command
does not require manually composing an `ncu` invocation.

## Correctness first

Use `compute-sanitizer` before trusting benchmark results if a kernel has touched raw indexing logic.

```bash
compute-sanitizer python3 path/to/test.py
```
