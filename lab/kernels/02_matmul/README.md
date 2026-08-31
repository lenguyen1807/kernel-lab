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

## Resources

- https://siboehm.com/articles/22/CUDA-MMM
- https://www.aleksagordic.com/blog/matmul#cpt1
- https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html
- https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
- https://github.com/gau-nernst/learn-cuda/tree/main/02a_matmul_simt
