# Profiling Fundamentals: Vector Add Case Study

Profiling is not looking through counters until an optimization appears. A
profiler supplies observations; an analytical model and a falsifiable
hypothesis turn those observations into an explanation.

The loop is:

1. model the work;
2. predict the bottleneck, fraction of peak, and variant ordering;
3. benchmark without profiler interference;
4. identify one discrepancy between prediction and measurement;
5. collect evidence that distinguishes competing explanations;
6. inspect generated code for a concrete compiler question;
7. change one thing and repeat.

This records the first application of that loop to `01_vecadd`.

## The observation stack

| Question | Instrument |
| --- | --- |
| Is it correct? | Tests and `compute-sanitizer` |
| How fast is it? | CUDA-event benchmark |
| What launched, when, and with what gaps? | Nsight Systems |
| What limits one kernel? | Nsight Compute |
| What did the compiler emit? | `ptxas`, PTX, SASS, `cuobjdump`, `nvdisasm` |

Nsight Systems is a timeline tool. Nsight Compute collects per-kernel counters.
PTX/SASS inspection answers narrower compiler questions. Do not ask one layer
to answer a question owned by another.

## Model before measurement

For fp32 vector addition:

```text
c[i] = a[i] + b[i]

FLOPs = N
useful bytes = 4 * (read A + read B + write C) * N = 12N
arithmetic intensity = N / 12N = 1/12 FLOP/byte
```

The arithmetic intensity is far below a modern GPU's ridge point, so the
predicted bottleneck is memory bandwidth.

### Arithmetic intensity is not bandwidth efficiency

```text
arithmetic intensity = FLOPs / bytes              [FLOP/byte]
measured bandwidth   = useful bytes / runtime     [GB/s]
HBM efficiency       = measured / peak bandwidth [dimensionless]
```

If `N = 1,000,000` and runtime is `0.1 s`:

```text
useful bytes = 12,000,000 bytes = 0.012 GB
bandwidth    = 0.012 GB / 0.1 s = 0.12 GB/s
```

That is not an efficiency until it is divided by peak bandwidth.

## Recorded hypothesis

```text
expected bottleneck: memory bandwidth
expected HBM efficiency: 1%
expected ordering: torch.add ~= scalar CUDA ~= float4 CUDA ~= Triton
compiler hypothesis: scalar remains scalar; float4 emits wider instructions
```

The 1% estimate was kept after correcting the units. Predictions must be
recorded before measurement; silently repairing them afterward destroys the
experiment.

A Triton program instance is not a CUDA thread block, and
`tl.arange(0, BLOCK_SIZE)` does not promise one CUDA thread per element. Triton
builds a vector of program values; its compiler maps them onto warps and
instructions.

## Correctness and benchmark

Correctness passed at `N = 1`, `100`, `1024`, and `1,000,003`. The odd
size exercises the vector kernel's remainder path.

Measured useful bandwidth on an NVIDIA A100 80GB PCIe:

| N | torch.add | scalar CUDA | float4 CUDA | Triton |
| ---: | ---: | ---: | ---: | ---: |
| 1,048,576 | 877.7 | 877.7 | 877.7 | 877.7 |
| 2,097,152 | 1,117.1 | 1,024.0 | 1,117.1 | 1,117.1 |
| 4,194,304 | 1,293.5 | 1,228.8 | 1,293.5 | 1,328.4 |
| 8,388,608 | 1,467.2 | 1,404.3 | 1,467.2 | 1,467.2 |
| 16,777,216 | 1,585.5 | 1,536.0 | 1,585.5 | 1,585.5 |
| 33,554,432 | 1,652.2 | 1,611.5 | 1,645.3 | 1,652.2 |

Values are GB/s. The full data is in
[`results/bench/vecadd/bench.csv`](../results/bench/vecadd/bench.csv).

At the largest shape:

```text
1652.2 / 1935 ~= 85.4% of nominal A100 PCIe bandwidth
```

The 1% prediction was wrong by roughly 85x. The ordering was closer: three
variants converged near 1650 GB/s, while scalar CUDA was about 2.5% behind.

## Why throughput rises with N

The numerator `12N` growing is not sufficient: runtime grows too. Changing
runtime from milliseconds to seconds cannot change its scaling.

Runtime grew about 17x while `N` grew 32x. A useful model is

```text
T(N) ~= T0 + 12N / B
bandwidth(N) = 12N / (T0 + 12N/B)
```

For small inputs, fixed cost `T0` matters. As `N` grows, it is amortized and
measured bandwidth approaches sustained streaming bandwidth `B`. The model
explains the trend but does not decompose `T0`.

## Coalescing is not vectorization

The scalar kernel can already be coalesced. Adjacent lanes request adjacent fp32
elements, so a warp's scalar load requests 128 contiguous bytes.

`float4` changes instruction width per lane:

```text
scalar lane:  4 bytes per load/store instruction
float4 lane: 16 bytes per load/store instruction
```

It does not change useful bytes and ideally does not change total memory sectors.
Its proposed benefit is fewer dynamic warp-level memory and address instructions.
Equal runtime therefore does not imply equal generated code: distinct
implementations can converge on the same HBM bottleneck.

## PTX is not the bottom

```text
CUDA C++ or GPU DSL
        |
        v
       PTX       virtual instruction set
        |
        | ptxas
        v
       SASS      machine instructions for a target SM
        |
        v
       GPU
```

PTX helps explain compiler transformations and DSL code generation. SASS is the
final authority for instructions executed from a cubin. This repo uses
`-lineinfo` for source-to-SASS mapping and `-Xptxas=-v` for register/shared
memory reporting.

Inspect assembly only for a concrete question: vector load width, unrolling,
FMA generation, address arithmetic, or spills.

## Scalar versus float4 SASS

The `sm_80` disassembly confirmed the compiler hypothesis.

```text
scalar hot path:       float4 hot path:

LDG.E                  LDG.E.128
LDG.E                  LDG.E.128
FADD                   FADD x4
STG.E                  STG.E.128
```

For 128 output elements:

```text
scalar:
    4 warps
    12 memory warp-instructions
    4 arithmetic warp-instructions

float4:
    1 warp
    3 memory warp-instructions
    4 arithmetic warp-instructions
```

Both move 1536 useful bytes. Vectorization reduces dynamic memory warp
instructions by about 4x without reducing data volume.

| Kernel | Registers/thread | Stack | Shared | Local |
| --- | ---: | ---: | ---: | ---: |
| scalar | 12 | 0 | 0 | 0 |
| float4 | 20 | 0 | 0 | 0 |

Neither spills. More registers are a tradeoff, not automatically a problem;
occupancy analysis must show whether residency is actually reduced.

The larger `float4` disassembly includes scalar remainder cleanup. Benchmark
shapes are divisible by four, so that loop is not the hot path. Assembly must be
interpreted with the input shape and branch conditions.

## What is proved and what is not

Supported:

- the cost model predicts a memory-bound kernel;
- large variants reach about 85% of nominal HBM bandwidth;
- scalar CUDA is consistently slightly slower after the smallest shape;
- scalar and `float4` emit different-width memory instructions;
- `float4` executes fewer memory warp-instructions per output;
- neither CUDA variant spills.

Not yet proved:

> Scalar CUDA is instruction-throughput-bound.

Extra instructions can be overhead without being the primary bottleneck. Both
kernels can remain HBM-bound while scalar instruction pressure slightly reduces
how closely it approaches the memory roof.

```text
T ~= max(T_memory, T_instruction, T_other)
```

If `T_memory` dominates both variants, fewer instructions need not improve
latency.

## What Nsight Compute must distinguish

| Observation | Same HBM ceiling | Scalar instruction pressure limits saturation |
| --- | --- | --- |
| DRAM throughput | Both similarly near peak | Scalar lower; float4 closer |
| DRAM sectors/output | Approximately equal | Still approximately equal |
| Instructions/output | Different but hidden | Scalar higher and correlated with slowdown |
| Issue pipeline | Not saturated | Relevant issue/LSU pressure elevated |
| Warp states | Memory latency may appear | `LG Throttle` may increase |
| Vectorization | Little speedup | Fewer instructions enable more HBM throughput |

No single counter proves causality. The inference requires agreement among the
controlled A/B timing, sectors, instructions, and pipeline behavior.

Read a report top-down:

1. verify kernel, shape, launch dimensions, and duration;
2. use Speed of Light/roofline to locate memory versus compute pressure;
3. inspect Memory Workload Analysis and sectors;
4. inspect Launch Statistics and the resource limiting occupancy;
5. inspect scheduler issue activity and eligible warps;
6. inspect warp stalls only if schedulers fail to issue enough work;
7. correlate CUDA, PTX, and SASS on the Source page.

Occupancy is not utilization or a performance score. A large warp-stall
percentage is not a diagnosis. `Long Scoreboard` can be normal for dependent
memory operations, while `LG Throttle` matters only in context.

## How to read Nsight Compute

### What “GPU SOL” means

SOL means Speed of Light: NCU’s estimate of a hardware unit’s sustainable ceiling.

A typical SOL percentage is approximately:

$$
\text{SOL\%} = \frac{\text{measured rate during the kernel}}
{\text{NCU peak sustained rate}}
\times 100
$$

The important words are:

- **Rate**, not total work.
- **Peak sustained**, not necessarily the marketing theoretical peak or a one-cycle burst.
- During elapsed kernel time, including cycles where that unit was idle.
- Often a composite metric: the maximum utilization among several constituent counters.

The full metric behind “Compute (SM) Throughput,” for example, is shaped like:

```text
sm__throughput.avg.pct_of_peak_sustained_elapsed
│               │   │                        │
unit            rollup normalization         time basis
```

That decodes as:
- **sm**: measured on the streaming multiprocessors.
- **throughput**: a composite of relevant SM pipelines.
- **avg**: averaged across SM instances.
- **pct_of_peak_sustained**: normalized to NCU’s sustained peak model.
- **elapsed**: divided by the entire elapsed interval, not only cycles where the unit was active.

Therefore:

> “Memory Throughput = 87.33%” does not mean every memory subsystem is 87.33% busy.

It means the busiest relevant memory constituent reached 87.33% of its sustained peak.

### First learn the metric grammar

When you see any NCU number, ask five mechanical questions:

1. What hardware unit?
    sm, smsp, l1tex, lts/L2, dram, etc.

2. What is counted?
    Bytes, sectors, instructions, active cycles, requests, warps?

3. How is it aggregated?
    sum, avg, min, or max across hardware instances?

4. What is the time basis?
    active counts only cycles when the unit is active; elapsed includes the complete measurement
    interval.

5. Is it absolute or normalized?
    Bytes, bytes/s, instructions/cycle, or percentage of peak?

Do not interpret a percentage until those five answers are clear.

> Read more at [metrics guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-guide)

### Read the launch identity first

The scalar vecadd report header contains

```text
Size:       (131072, 1, 1) x (256, 1, 1)
Time:       243.65 us
Cycles:     258,665
SM clock:   1.06 GHz
Registers:  16/thread
```

The left size tuple is the grid and the right tuple is the block:

```text
131,072 blocks x 256 threads/block
```

For the scalar kernel,

```text
131,072 x 256 = 33,554,432 = N
```

so this is the intended largest shape with one thread per element. Verify the
kernel name, shape, grid, and block before interpreting performance counters. A
report for the wrong launch answers the wrong question precisely.

The duration and cycle count express approximately the same interval in
different units:

```text
243.648 us x 1.061634 GHz ~= 258,665 cycles
```

Nsight Compute controls clocks and may replay the kernel to collect incompatible
counters. Use this duration to compare reports collected with the same settings,
not as a replacement for the ordinary benchmark.

`16 registers/thread` is a static resource allocation for this compiled kernel.
It does not by itself show that registers limit performance. That requires
combining registers per thread with threads per block, resident blocks per SM,
and the occupancy limits reported later.

### Decode the vecadd SOL table literally

The scalar report shows

| Metric | Value | Literal reading |
| --- | ---: | --- |
| Memory Throughput | 87.33% | Highest normalized throughput among the selected memory constituents |
| DRAM Throughput | 87.33% | Highest normalized throughput among the selected DRAM constituents |
| L2 Throughput | 82.85% | Highest normalized throughput among the selected L2 constituents |
| L1/TEX Throughput | 20.20% | Highest normalized throughput among the selected L1/TEX constituents |
| Compute (SM) Throughput | 15.11% | Highest normalized throughput among the selected SM execution constituents |

The high-level Memory Throughput equals DRAM Throughput because DRAM is the
largest constituent in this report:

```text
max(DRAM 87.33%, L2 82.85%, L1/TEX 20.20%, ...) = 87.33%
```

This table does not contain cache hit rates. Throughput and hit rate have
different denominators:

```text
cache throughput = transferred work / capacity per unit time
cache hit rate    = cache hits / cache lookups
```

A streaming kernel can have high L2 throughput without temporal reuse: data is
passing through L2 quickly, not necessarily being found there repeatedly.

Likewise, Compute (SM) Throughput is not the percentage of peak useful fp32
FLOP/s. It is a composite over several SM execution pipelines, including
load/store, address, arithmetic, and control instructions. In this report the
load/store instruction constituent is about 15.09%, which nearly determines the
15.11% composite. Expand **GPU Throughput Breakdown** before assigning a
meaning to the headline number.

### Useful bandwidth and hardware bandwidth are different

Nsight Compute reports approximately 1.69 TB/s at the DRAM interface. Dividing
that by the A100 PCIe's nominal 1.935 TB/s gives the reported SOL value:

```text
1.69 / 1.935 ~= 87.3%
```

The analytical model uses useful algorithmic bytes instead:

```text
useful bytes = 12N
             = 12 x 33,554,432
             = 402,653,184 bytes

useful bandwidth = 402,653,184 bytes / 243.648 us
                 ~= 1.653 TB/s
```

Both numbers are valid, but their numerators differ:

- useful bandwidth counts the `12N` bytes required by the algorithm;
- hardware DRAM bandwidth counts transactions observed at the DRAM interface.

Transaction-level traffic can differ from the useful-byte model. Do not silently
substitute one bandwidth for the other; compare them and investigate a material
gap using sector and byte counters.

### Activity, throughput, and occupancy are different

The report contains approximately

```text
elapsed cycles:   258,665
SM active cycles: 254,330 average
```

so the SMs were active for roughly

```text
254,330 / 258,665 ~= 98.3%
```

An active SM has resident work. It need not issue an instruction every cycle,
and its arithmetic pipelines need not be busy. An SM can remain active while
all its resident warps wait for memory. Therefore these measurements are
compatible:

```text
SM active cycles       ~= 98% of elapsed cycles
Compute (SM) Throughput = 15.11% of sustained peak
```

Neither value is occupancy. Occupancy is the number of resident warps relative
to the hardware's residency limit; activity says whether work is present;
throughput says how quickly a hardware unit is processing work.

The report also shows `151.70 Waves Per SM`. With 256-thread blocks, each block
contains eight warps. A GA100 SM can hold 64 warps, so the warp limit permits
eight such blocks per SM. Across 108 SMs, one full wave is approximately

```text
108 SMs x 8 blocks/SM = 864 blocks
```

and the launch contains

```text
131,072 blocks / 864 blocks/wave ~= 151.70 waves
```

This says the grid is much larger than the GPU's instantaneous residency, so
the final partial wave is a small fraction of the launch. Waves per SM is not an
occupancy percentage.

### Stop at observation before claiming causality

From the SOL section alone, the supported observations are

- DRAM is the most highly utilized reported subsystem at 87.33%;
- L2 is also highly utilized at 82.85%;
- no reported SM execution constituent exceeds about 15.11%;
- the DRAM interface transfers approximately 1.69 TB/s.

The SOL section alone does not establish that Long Scoreboard, occupancy, or
instruction pressure limits performance. Those claims require the memory,
scheduler, warp-state, and instruction sections. The key sanity question is:

> How can an SM be active for 98% of elapsed cycles while its Compute
> Throughput is only 15%?

The answer is the distinction between having resident work and issuing useful
work at a high rate. That distinction is required before interpreting occupancy
or warp stalls.


## Measurement discipline

Use `bench` for performance and `profile` for explanations. Nsight Compute
may replay kernels because all counters cannot be collected in one pass. Clock
control and cache flushing also make profiled durations differ from ordinary
benchmarks.

Profile one representative shape. The order remains:

```text
test -> compute-sanitizer after indexing changes -> bench -> profile
```

## Environment result

The attempted collection did not reach GPU counters:

1. `nsight-python` required Nsight Compute 2026.2.1+, but the server exposed
   2025.1.1;
2. direct `ncu` returned `ERR_NVGPUCTRPERM` because the driver restricts
   performance counters to privileged processes.

These are profiling-plumbing failures, not kernel evidence. The repo's
`kernel_lab_profile` wrapper handles the permission case; tool versions must also
be compatible.

## References

- [Simon Boehm's CUDA matmul worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [NVIDIA CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)
- [Profiling mechanics for this repo](profiling.md)
- [Roofline notes](roofline.md)
