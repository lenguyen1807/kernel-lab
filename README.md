# Kernel Lab

A personal lab for learning to write GPU kernels for LLM inference — CUDA
today, ROCm/FlyDSL when an AMD box enters the picture, with Python kernel DSLs
(Triton, Tilelang, CuTeDSL, Helion, Gluon) as first-class variants alongside
the handwritten CUDA.

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
uv run kernel-lab test  02_matmul    # are the kernels correct?
uv run kernel-lab bench 02_matmul    # how fast are they?
```

`bench` prints one table per shape:

```text
### 2048x2048x2048

| Kernel        |      ms | peak mem | TFLOP/s |   GB/s | vs torch | sdpa |
| ------------- | ------: | -------: | ------: | -----: | -------: | ---: |
| torch         |  ...    |    ...   |   ...   |  ...   |    1.00x |   -  |
| cublas        |  ...    |    ...   |   ...   |  ...   |    ...   |   -  |
| naive         |  ...    |    ...   |   ...   |  ...   |    ...   |   -  |
| tiled         |  ...    |    ...   |   ...   |  ...   |    ...   |   -  |
```

(Layout only — fill in your own numbers, and record the GPU alongside them.)

`peak mem` is the extra device memory one call allocates above its live
inputs — the S/P-materialization line a fused kernel exists to erase.
`sdpa` names the fused backend that actually serviced a `torch.sdpa` call
(`flash` / `mem_efficient` / `cudnn` / `math`), `-` when the variant never
calls sdpa: the dispatcher's choice is otherwise invisible in the result.
A variant that OOMs at a shape is recorded (`OOM` in the table, `status=oom`
in the CSV) and skipped at all larger shapes, so one fat shape no longer
kills the sweep. Allocator near-misses — a `cudaMalloc` that failed, evicted
the cache, and was retried (the `W921` stderr warning) — are starred in the
table and counted in `alloc_retries`; the allocation still succeeded.

Add `--plot` for latency-vs-size and peak-memory-vs-size curves under
`results/bench/<kernel>/`.

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

**Use `bench` for numbers and `profile` for explanations.** `profile` runs
each variant once under `ncu --set full` at a single shape and writes one
`.ncu-rep` per variant to `results/profiles/<kernel>/` — copy the files
elsewhere and open them in the Nsight Compute GUI (`ncu-ui`) to inspect
occupancy, warp stalls, and memory throughput interactively. It is the wrong
tool for drawing a curve — that is what `bench` is for.

```bash
uv run kernel-lab bench   02_matmul --nsight   # same sweep, Nsight timing
uv run kernel-lab profile 02_matmul            # counters at the largest shape
uv run kernel-lab profile 02_matmul --shape 1024
```

Nsight locks clocks to base while profiling, so durations inside a report look
slower than `bench` numbers — compare ratios and bottleneck sections, not raw
milliseconds.

On hosts where the driver restricts performance counters to root, use the
wrapper from the setup script, which keeps the venv and fixes file ownership
afterwards:

```bash
kernel_lab_profile 02_matmul
```

## Repo map

```text
lab/harness.py            The whole harness: build, test, bench, profile.
lab/cli.py                Finds lab/kernels/<kernel>/main.py and runs it.
lab/kernels/utils.h       Shared CUDA macros (CHECK_INPUT, CUDA_CHECK, ...).
lab/kernels/01_vecadd/    Memory-bound baseline: CUDA, float4, Triton.
lab/kernels/02_matmul/    fp32 SIMT ladder: naive → tiled → 1D → 2D.
lab/kernels/02_matmul_pmpp/  Same algorithms, PMPP indexing idiom.
lab/kernels/03_matmul_sm80/  Tensor-core track: bf16, B K-major, %SOL.
docs/                     Profiling, roofline, and machine notes.
results/bench/<kernel>/   bench_<gpu>.csv, bench_<gpu>.png — one set per machine
results/profiles/<kernel>/  Nsight CSVs, charts, .ncu-rep (GPU-tagged)
```

## Backends

One torch-track harness: a variant is any callable over torch tensors, so
Triton, Tilelang, CuTeDSL, Helion, and Gluon all drop into the same `VARIANTS`
dict — a new DSL is a new subfolder in the kernel directory, not a new harness.
On a ROCm box the same `main.py` works (ROCm torch exposes the CUDA API
surface, device type included), but compiled kernels are native per vendor —
`cuda/` for NVIDIA, `hip/` for AMD, no hipify in between:

```python
ext = harness.load_kernel(__file__, source_dir="hip" if torch.version.hip else "cuda")
```

`load_kernel` compiles `*.hip` with hipcc directly; torch only treats `.hip`
as a GPU source on ROCm builds, so the two dirs can never be mixed up by
accident. Register vendor-specific variants — inline PTX, CuTeDSL, MFMA —
conditionally:

```python
if torch.version.hip is None:
    VARIANTS["cutedsl"] = ...
```

Environments are per machine, via uv groups: core deps everywhere,
`--group cuda` on the NVIDIA box (`setup_cuda_env.sh` does it), `--group rocm`
once that track exists. `profile` is Nsight-only until the rocprof backend
lands with the ROCm track.

## Adding a kernel

```bash
cp -r lab/kernels/01_vecadd lab/kernels/03_reduction
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
- **Dtype and layout are folder-level contracts, not per-variant choices.** A
  table only means something when every row computes the same problem in the
  same precision — an fp16 tensor-core kernel in an fp32 folder measures the
  hardware peak gap, not kernel quality. A variant is `f(*inputs) -> out`
  with no casts or allocations inside the timed call.
- **`sol=` buys honesty.** Pass the spec-sheet peak for the folder's dtype
  (scalar or `{gpu-name-substring: tflops}`) and `bench` adds a %SOL column;
  `num_input_sets>1` rotates distinct inputs across reps so nothing is
  bitwise-identical or L2-resident. `bench` also verifies each variant
  against `ref` before timing it (`check=False` to opt out).
- **Never `torch.cuda.synchronize()` inside a variant.** It serialises the
  benchmark loop and quietly inflates that one row.
- **Allocate with `torch.empty`, not `torch.zeros`**, when the kernel overwrites
  its output — `zeros` launches an extra fill kernel that shows up in profiles.
- Kernel sources are compiled with `-lineinfo` (SASS maps back to source) and
  `-Xptxas=-v` (per-kernel register and shared-memory usage in the build log).
- **Result files are per-machine**: `bench` writes `bench_<gpu>.csv` /
  `bench_<gpu>.png`, so sweeps from different GPUs coexist in git instead of
  clobbering each other. Cite the GPU in the kernel README's measured table.
- **On ROCm, torch hipifies `.cu` sources and builds them with hipcc** — the
  build is not what breaks. What breaks is inline PTX and 32-wide-warp
  assumptions (wavefronts are 64-wide on CDNA, e.g. MI300X; 32-wide is only
  the RDNA default).

## Acknowledgements

Inspired by [Gau Nernst's `learn-cuda`](https://github.com/gau-nernst/learn-cuda),
especially the one-`main.py`-per-kernel layout and benchmarking through Triton's
`do_bench`. Matmul variants follow [Simon Boehm's
worklog](https://siboehm.com/articles/22/CUDA-MMM) and *Programming Massively
Parallel Processors*.
