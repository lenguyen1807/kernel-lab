# Matrix Multiplication

```bash
uv run kernel-lab test  02_matmul
uv run kernel-lab bench 02_matmul --plot
```

## Variants

| Name | Idea | Expected limit |
| --- | --- | --- |
| `torch` | whatever `torch.matmul` dispatches to | reference |
| `cublas` | explicit `cublasSgemm` | reference |
| `naive` | one thread per output element | global memory: re-reads a full row and column per element |
| `tiled` | 32x32 shared-memory tiling | shared memory traffic; one output per thread |
| `1D_coarsening` | each thread owns a `TM x 1` strip | register/ILP bound, better reuse |
| `2D_coarsening` | each thread owns a `TM x TN` patch | closest to compute bound |

`naive` is skipped above 2048 (`limits` in `main.py`) — past that it costs more
wall clock than it teaches.

> Indexing walkthrough for the coarsened kernels: [`BLOCK_TILING.md`](BLOCK_TILING.md).
> A PMPP-style version of the same algorithms lives in
[`../02_matmul_pmpp/`](../02_matmul_pmpp/).

## Cost model

For square `N`:

```text
FLOPs = 2 N^3                       one multiply + one add per (m, n, k)
bytes = 4 (2 N^2 + N^2) = 12 N^2    read A and B once, write C once (perfect reuse)
AI    = 2 N^3 / 12 N^2 = N / 6      flops per byte
```

So arithmetic intensity grows linearly with `N`: matmul is compute bound at any
interesting size, and every variant here is really a story about *getting close
to that ideal byte count*. `naive` moves `4 (2 N^3 + N^2)` bytes instead —
`N/6` collapses to about `1/4`, squarely memory bound. That single ratio is why
tiling exists.

## Results

```
GPU: NVIDIA RTX PRO 4500 Blackwell | torch 2.14.0+cu130 | CUDA 13.0

### 512x512x512

| Kernel        |     ms | TFLOP/s |  GB/s | vs torch |
| ------------- | -----: | ------: | ----: | -------: |
| torch         | 0.0236 |   11.40 | 133.6 |    1.00x |
| cublas        | 0.0225 |   11.93 | 139.8 |    1.05x |
| naive         | 0.1021 |    2.63 |  30.8 |    0.23x |
| tiled         | 0.0857 |    3.13 |  36.7 |    0.27x |
| 1D_coarsening | 0.0601 |    4.47 |  52.3 |    0.39x |
| 2D_coarsening | 0.1103 |    2.43 |  28.5 |    0.21x |

### 1024x1024x1024

| Kernel        |     ms | TFLOP/s |  GB/s | vs torch |
| ------------- | -----: | ------: | ----: | -------: |
| torch         | 0.0881 |   24.38 | 142.8 |    1.00x |
| cublas        | 0.0881 |   24.38 | 142.8 |    1.00x |
| naive         | 0.6336 |    3.39 |  19.9 |    0.14x |
| tiled         | 0.5199 |    4.13 |  24.2 |    0.17x |
| 1D_coarsening | 0.2209 |    9.72 |  57.0 |    0.40x |
| 2D_coarsening | 0.2004 |   10.72 |  62.8 |    0.44x |

### 2048x2048x2048

| Kernel        |     ms | TFLOP/s | GB/s | vs torch |
| ------------- | -----: | ------: | ---: | -------: |
| torch         | 0.5320 |   32.29 | 94.6 |    1.00x |
| cublas        | 0.5468 |   31.42 | 92.0 |    0.97x |
| naive         | 5.1886 |    3.31 |  9.7 |    0.10x |
| tiled         | 3.8622 |    4.45 | 13.0 |    0.14x |
| 1D_coarsening | 1.4036 |   12.24 | 35.9 |    0.38x |
| 2D_coarsening | 1.1240 |   15.28 | 44.8 |    0.47x |

### 4096x4096x4096

| Kernel        |      ms | TFLOP/s | GB/s | vs torch |
| ------------- | ------: | ------: | ---: | -------: |
| torch         |  4.5200 |   30.41 | 44.5 |    1.00x |
| cublas        |  4.5466 |   30.23 | 44.3 |    0.99x |
| naive         | skipped |         |      |          |
| tiled         | 31.7041 |    4.34 |  6.4 |    0.14x |
| 1D_coarsening | 11.3710 |   12.09 | 17.7 |    0.40x |
| 2D_coarsening |  7.7348 |   17.77 | 26.0 |    0.58x |
```

## Resources

- https://siboehm.com/articles/22/CUDA-MMM
- https://www.aleksagordic.com/blog/matmul#cpt1
- https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html
- https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
- https://github.com/gau-nernst/learn-cuda/tree/main/02a_matmul_simt
