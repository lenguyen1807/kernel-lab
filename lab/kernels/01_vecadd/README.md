# Vector Add

`c = a + b` in fp32. The simplest possible kernel, and the one where the
roofline argument is impossible to dodge.

```bash
uv run kernel-lab test  01_vecadd
uv run kernel-lab bench 01_vecadd --plot
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

`cuda_float4` exists to test one hypothesis: does a 128-bit access per thread
close the gap to peak? See [the NVIDIA post on vectorized memory access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/).

## Results

```text
### 1048576

| Kernel      |     ms | TFLOP/s |  GB/s | vs torch.add |
| ----------- | -----: | ------: | ----: | -----------: |
| torch.add   | 0.0143 |    0.07 | 877.7 |        1.00x |
| cuda        | 0.0143 |    0.07 | 877.7 |        1.00x |
| cuda_float4 | 0.0143 |    0.07 | 877.7 |        1.00x |
| triton      | 0.0143 |    0.07 | 877.7 |        1.00x |

### 2097152

| Kernel      |     ms | TFLOP/s |   GB/s | vs torch.add |
| ----------- | -----: | ------: | -----: | -----------: |
| torch.add   | 0.0225 |    0.09 | 1117.1 |        1.00x |
| cuda        | 0.0246 |    0.09 | 1024.0 |        0.92x |
| cuda_float4 | 0.0225 |    0.09 | 1117.1 |        1.00x |
| triton      | 0.0225 |    0.09 | 1117.1 |        1.00x |

### 4194304

| Kernel      |     ms | TFLOP/s |   GB/s | vs torch.add |
| ----------- | -----: | ------: | -----: | -----------: |
| torch.add   | 0.0379 |    0.11 | 1328.4 |        1.00x |
| cuda        | 0.0420 |    0.10 | 1198.8 |        0.90x |
| cuda_float4 | 0.0389 |    0.11 | 1293.5 |        0.97x |
| triton      | 0.0389 |    0.11 | 1293.5 |        0.97x |

### 8388608

| Kernel      |     ms | TFLOP/s |   GB/s | vs torch.add |
| ----------- | -----: | ------: | -----: | -----------: |
| torch.add   | 0.0686 |    0.12 | 1467.2 |        1.00x |
| cuda        | 0.0717 |    0.12 | 1404.3 |        0.96x |
| cuda_float4 | 0.0686 |    0.12 | 1467.2 |        1.00x |
| triton      | 0.0686 |    0.12 | 1467.2 |        1.00x |

### 16777216

| Kernel      |     ms | TFLOP/s |   GB/s | vs torch.add |
| ----------- | -----: | ------: | -----: | -----------: |
| torch.add   | 0.1260 |    0.13 | 1598.4 |        1.00x |
| cuda        | 0.1311 |    0.13 | 1536.0 |        0.96x |
| cuda_float4 | 0.1270 |    0.13 | 1585.5 |        0.99x |
| triton      | 0.1270 |    0.13 | 1585.5 |        0.99x |

### 33554432

| Kernel      |     ms | TFLOP/s |   GB/s | vs torch.add |
| ----------- | -----: | ------: | -----: | -----------: |
| torch.add   | 0.2437 |    0.14 | 1652.2 |        1.00x |
| cuda        | 0.2499 |    0.13 | 1611.5 |        0.98x |
| cuda_float4 | 0.2447 |    0.14 | 1645.3 |        1.00x |
| triton      | 0.2437 |    0.14 | 1652.2 |        1.00x |
```