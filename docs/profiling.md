# Profiling

`bench` tells you *how fast*. `profile` tells you *why*. This note is about the
second one.

## When to reach for Nsight

Only after `cuda-lab bench` has told you which kernel is worth explaining, and
only when you need something latency cannot give you:

- achieved vs peak memory throughput,
- occupancy, and what is limiting it (registers, shared memory, block size),
- warp stall reasons,
- shared-memory bank conflicts,
- source-level hot spots (kernels are compiled with `-lineinfo`).

Do **not** use it to draw a latency-vs-size curve. Nsight replays every
annotated region once per metric pass, flushing caches and locking clocks in
between, so its cost scales with `shapes × variants × runs` no matter how fast
the kernel is. A sweep that `bench` finishes in seconds takes minutes here.

## Prerequisites

- A CUDA-capable NVIDIA GPU on Linux.
- Nsight Compute (`ncu`) on `PATH`.
- `nsight-python`, declared as a Linux-only dependency so macOS can still edit
  the lab. It drives Nsight Compute; it does not bundle it.

## Running

```bash
uv run cuda-lab profile 02_matmul                # counters at the largest shape
uv run cuda-lab profile 02_matmul --shape 1024   # or pick one
uv run cuda-lab bench   02_matmul --nsight       # whole sweep, Nsight-timed
```

If the driver restricts performance counters to privileged processes, use the
helper from `scripts/setup_cuda_env.sh` — it preserves the venv and CUDA
environment under `sudo`, then hands the artifacts back to you:

```bash
source scripts/setup_cuda_env.sh
cuda_lab_profile 02_matmul
```

Output lands in `results/profiles/<kernel>/`: aggregated CSV, PNG, the
`.ncu-rep` report, and a log. Open the `.ncu-rep` in the Nsight Compute UI —
that is where the counters actually become readable.

## How the harness drives Nsight

`harness._nsight_sweep` wraps your `VARIANTS` dict in the nsight-python
decorator pair:

- `@nsight.analyze.kernel(configs=...)` — one entry per shape.
- `nsight.annotate(label)` — one region per launch; each label becomes a series.
- `@nsight.analyze.plot(...)` — line chart, added only for `--nsight` sweeps.

Two settings are load-bearing:

- `replay_mode="range"` with `combine_kernel_metrics=lambda x, y: x + y`,
  because `torch.matmul` and cuBLAS may launch several kernels per call and
  their durations need to sum.
- `clock_control="base"`, which makes runs comparable but slower than reality.
  This is why Nsight and `bench` disagree on absolute numbers.

Keep annotated regions to one kernel launch each where you can, and allocate
outputs with `torch.empty` rather than `torch.zeros` — `zeros` launches a fill
kernel that pollutes the region. `lab/kernels/utils.h` provides `create_matrix`
for exactly this reason.

`ncu --query-metrics` on the CUDA machine lists metrics you can add to a profile
script. You should not need to hand-write an `ncu` command line.

## Correctness first

A fast wrong kernel is worth nothing. After touching any indexing logic:

```bash
compute-sanitizer python lab/kernels/02_matmul/main.py test
```
