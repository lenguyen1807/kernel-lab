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

Each profiled kernel folder provides a `profile.py` script. For example, profile
the vector-add variants with:

```bash
uv run cuda-lab profile 01_vecadd
```

The script uses `@nsight.analyze.kernel` to manage the profiling session and
`nsight.annotate(...)` to identify each kernel launch. It writes the raw and
aggregated CSVs, Nsight Compute report, and log under
`results/profiles/01_vecadd/`, then prints the aggregated table.

Add a `profile.py` alongside a new kernel's `bench.py`. Annotate exactly one
kernel launch per region. If an annotation intentionally launches several
kernels, set `replay_mode="range"` or provide `combine_kernel_metrics` so the
results have an unambiguous interpretation.

Use `ncu --query-metrics` on the CUDA machine only when you need to discover
additional hardware metrics for a profile script. The standard `profile` command
does not require manually composing an `ncu` invocation.

## Correctness first

Use `compute-sanitizer` before trusting benchmark results if a kernel has touched raw indexing logic.

```bash
compute-sanitizer python3 path/to/test.py
```
