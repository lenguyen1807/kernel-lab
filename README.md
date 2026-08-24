# CUDA Lab

A personal lab for learning to write CUDA kernels for LLM inference.

Every kernel here exists twice: once as CUDA source, once as a written argument
for why it should be fast. The harness is deliberately small — its only job is
to make the compile → test → benchmark loop boring, so the interesting work
stays where it belongs: counting bytes, estimating rooflines, and explaining the
gap between what you predicted and what the GPU did.

This is not a kernel library. Do not depend on it.

## Quickstart

On the CUDA machine, after SSH:

```bash
source scripts/setup_cuda_env.sh     # toolchain, venv, sanity checks
uv run cuda-lab test  02_matmul      # are the kernels correct?
uv run cuda-lab bench 02_matmul      # how fast are they?
```

`bench` prints one table per shape:

```text
### 2048x2048x2048

| Kernel        |      ms | TFLOP/s |   GB/s | vs torch |
| ------------- | ------: | ------: | -----: | -------: |
| torch         |  ...    |   ...   |  ...   |    1.00x |
| cublas        |  ...    |   ...   |  ...   |    ...   |
| naive         |  ...    |   ...   |  ...   |    ...   |
| tiled         |  ...    |   ...   |  ...   |    ...   |
```

(Layout only — fill in your own numbers, and record the GPU alongside them.)

Add `--plot` for a latency-vs-size curve under `results/bench/<kernel>/`.

Each `main.py` also runs standalone, which is handy under a debugger or
`compute-sanitizer`:

```bash
python lab/kernels/02_matmul/main.py bench --plot
```

## The three commands

| Command | What it answers | Cost |
| --- | --- | --- |
| `test` | Is it correct? | instant |
| `bench` | How fast is it? | seconds |
| `profile` | *Why* is it that fast? | minutes |

**Use `bench` for numbers and `profile` for explanations.** Nsight Compute
replays every annotated region once per metric pass, with cache flushes and
locked clocks in between, so its cost scales with `shapes × variants × runs`
regardless of how quick your kernel is. It is the right tool for occupancy,
warp stalls, and memory throughput at *one* shape. It is the wrong tool for
drawing a curve — that is what `bench` is for.

If you do want Nsight-measured durations across the whole sweep, they are one
flag away, and they will take minutes:

```bash
uv run cuda-lab bench   02_matmul --nsight   # same sweep, Nsight timing
uv run cuda-lab profile 02_matmul            # counters at the largest shape
uv run cuda-lab profile 02_matmul --shape 1024
```

Expect `bench` and `bench --nsight` to agree on kernel *ordering* but not on
absolute numbers: Nsight locks clocks to base, so everything looks slower.

On hosts where the driver restricts performance counters to root, use the
wrapper from the setup script, which keeps the venv and fixes file ownership
afterwards:

```bash
cuda_lab_profile 02_matmul
```

`nsight-python` requires ncu ≥ 2026.2 (CUDA 13.3). On older toolkits, `profile`
and `bench --nsight` automatically fall back to driving the `ncu` CLI directly —
one process per (shape, variant), NVTX-filtered to a single launch. Same
artifacts, same locked clocks, a bit more process overhead.

## Repo map

```text
lab/harness.py            The whole harness: build, test, bench, profile.
lab/cli.py                Finds lab/kernels/<kernel>/main.py and runs it.
lab/kernels/utils.h       Shared CUDA macros (CHECK_INPUT, CUDA_CHECK, ...).
lab/kernels/00_template/  Copy this to start a kernel.
lab/kernels/01_vecadd/    Memory-bound baseline: CUDA, float4, Triton.
lab/kernels/02_matmul/    Simon Boehm's ladder: naive → tiled → 1D → 2D.
lab/kernels/02_matmul_pmpp/  Same algorithms, PMPP indexing idiom.
docs/                     Profiling, roofline, and machine notes.
results/bench/<kernel>/   bench.csv, bench.png
results/profiles/<kernel>/  Nsight CSVs, charts, .ncu-rep
```

## Adding a kernel

```bash
cp -r lab/kernels/00_template lab/kernels/03_reduction
```

A kernel folder is a `README.md`, a `main.py`, and a `cuda/` directory. The
harness compiles **every** `.cpp`/`.cc`/`.cu` file under `cuda/`, so a folder
can grow from `naive.cu` to `tiled.cu` to `vectorized.cu` without touching any
build config. One `ext.cpp` binds the launchers you want to reach from Python.

`main.py` is the whole Python side:

```python
import torch
from lab import harness

ext = harness.load_kernel(__file__)      # compiles cuda/*, caches by folder name

VARIANTS = {
    "torch": torch.matmul,               # the reference is a variant too
    "naive": ext.matmul_naive,
    "tiled": ext.matmul_tiled,
}

def make_inputs(m, k, n):
    return (torch.randn(m, k, device="cuda"), torch.randn(k, n, device="cuda"))

if __name__ == "__main__":
    harness.main(
        VARIANTS,
        make_inputs=make_inputs,
        shapes=[(n, n, n) for n in (512, 1024, 2048, 4096)],
        check_shapes=[(31, 17, 29), (128, 256, 64)],   # odd sizes catch guard bugs
        ref=torch.matmul,
        flops=lambda m, k, n: 2 * m * k * n,
        nbytes=lambda m, k, n: 4 * (m * k + k * n + m * n),
        limits={"naive": 2048},          # skip a variant once it is just slow
    )
```

`flops` and `nbytes` are what turn latency into TFLOP/s and GB/s. Write them
from the algorithm, not from the code — that is the number you compare your
measurement against.

## What a kernel README should answer

The folder README is the report. There is no separate reports directory: source,
benchmark, and writeup live together or they drift apart.

1. **Problem and baseline.** What is computed, and what are you racing?
2. **Variants and expected bottleneck.** Memory or compute, and why?
3. **FLOPs, bytes, arithmetic intensity.** Written down *before* measuring.
4. **Roofline expectation.** What fraction of peak should the best one reach?
5. **Measured table.** `bench` output, plus which GPU produced it.
6. **What failed, and what's next.** The section you will actually reread.

If you cannot state the expected bottleneck before running the kernel, you are
benchmarking without a hypothesis. See [`docs/roofline.md`](docs/roofline.md).

## Gotchas

- **Do not pin `TORCH_CUDA_ARCH_LIST`** unless you know why. Left unset, PyTorch
  detects the real compute capability. Pinned to the wrong value and without a
  `+PTX` suffix, the cubin will not load on your GPU at all.
- **Extension names come from the folder name**, so `02_matmul` and
  `02_matmul_pmpp` get separate build caches. Hardcoding the same name in two
  folders makes torch rebuild everything each time you switch between them.
- **Run `compute-sanitizer` after touching indexing**, before trusting any
  number: `compute-sanitizer python lab/kernels/02_matmul/main.py test`.
- **Never `torch.cuda.synchronize()` inside a variant.** It serialises the
  benchmark loop and quietly inflates that one row.
- **Allocate with `torch.empty`, not `torch.zeros`**, when the kernel overwrites
  its output — `zeros` launches an extra fill kernel that shows up in profiles.
- Kernel sources are compiled with `-lineinfo` (SASS maps back to source) and
  `-Xptxas=-v` (per-kernel register and shared-memory usage in the build log).

## Acknowledgements

Inspired by [Gau Nernst's `learn-cuda`](https://github.com/gau-nernst/learn-cuda),
especially the one-`main.py`-per-kernel layout and benchmarking through Triton's
`do_bench`. Matmul variants follow [Simon Boehm's
worklog](https://siboehm.com/articles/22/CUDA-MMM) and *Programming Massively
Parallel Processors*.
