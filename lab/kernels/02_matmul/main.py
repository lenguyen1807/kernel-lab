"""Matmul, following Simon Boehm's optimisation ladder.

naive -> shared-memory tiling -> 1D thread coarsening -> 2D thread coarsening,
against an explicit cublasSgemm call and whatever torch picks.
"""

from __future__ import annotations

import torch

from lab import harness
from tilelang_dsl import matmul_factory

ext = harness.load_kernel(__file__)

VARIANTS = {
    # torch may pick cuBLAS, cuBLASLt, or CUTLASS; `cublas` is always cublasSgemm.
    "torch": torch.matmul,
    "cublas": ext.matmul_cublas,
    "naive": ext.matmul_naive,
    "tiled": ext.matmul_tiled,
    "1D_coarsening": ext.matmul_1D_coarsening,
    "2D_coarsening": ext.matmul_2D_coarsening,
    "tilelang": harness.ShapeJIT(matmul_factory),
}


def make_inputs(m: int, k: int, n: int):
    return (
        torch.randn(m, k, device="cuda", dtype=torch.float32),
        torch.randn(k, n, device="cuda", dtype=torch.float32),
    )


if __name__ == "__main__":
    harness.main(
        VARIANTS,
        make_inputs=make_inputs,
        shapes=[(n, n, n) for n in (512, 1024, 2048, 4096)],
        check_shapes=[(16, 16, 16), (31, 17, 29), (128, 256, 64), (1024, 1024, 1024)],
        ref=torch.matmul,
        # 1 multiply + 1 add per (m, n, k) triple; read A and B, write C.
        flops=lambda m, k, n: 2 * m * k * n,
        nbytes=lambda m, k, n: 4 * (m * k + k * n + m * n),
        # naive re-reads a full row and column per output element -- past 2048 it
        # costs more wall time than it teaches.
        limits={"naive": 2048},
        # tilelang rounds its inputs to fp16; the rounding error accumulates
        # over the k-reduction, growing ~ sqrt(k) * 2^-11, plus a tail factor
        # from taking the max over m*n outputs (observed max abs diff: 0.037
        # at k=1024, 0.156 at k=4096). The fp32 kernels pass this easily.
        tol=lambda m, k, n: {"rtol": 2e-2, "atol": 3e-3 * k**0.5},
    )
