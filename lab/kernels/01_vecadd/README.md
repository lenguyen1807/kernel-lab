# Vector Add

`c = a + b` in fp32. The simplest possible kernel, and the one where the
roofline argument is impossible to dodge.

```bash
uv run cuda-lab test  01_vecadd
uv run cuda-lab bench 01_vecadd --plot
```

## Variants

| Name | Idea |
| --- | --- |
| `torch.add` | reference |
| `cuda` | one thread per element |
| `cuda_float4` | one thread per four elements via `float4`, 128-bit loads |
| `triton` | the Triton tutorial kernel, for comparison |

## Cost model

```text
FLOPs = N                one add per element
bytes = 4 * 3N           read a, read b, write c
AI    = 1 / 12           flops per byte
```

An arithmetic intensity of 1/12 is far below any GPU's ridge point, so **this
kernel is bandwidth bound and nothing will change that.** The only question
worth asking is what fraction of peak HBM bandwidth each variant reaches.
Compare the `GB/s` column against your GPU's spec sheet; TFLOP/s is noise here.

`cuda_float4` exists to test one hypothesis: does a 128-bit access per thread
close the gap to peak? See
[the NVIDIA post on vectorized memory access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/).

## Results

_Run `cuda-lab bench 01_vecadd` and paste the tables here, with the GPU name._

## What failed / next

_Record it here._
