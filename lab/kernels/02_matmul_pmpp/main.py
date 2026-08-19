"""Matmul written in PMPP style: full global indices instead of shifted pointers.

Same algorithms as 02_matmul, different indexing idiom. See README.md for the
side-by-side. Kept separate so both idioms stay readable.
"""

from __future__ import annotations

import torch

from lab import harness

ext = harness.load_kernel(__file__)

VARIANTS = {
    "torch": torch.matmul,
    "naive": ext.matmul_naive,
    "tiled_16": ext.matmul_tiled_16,
    "tiled_32": ext.matmul_tiled_32,
    "coarsening_16x4": ext.matmul_coarsening_16x4,
    "coarsening_32x4": ext.matmul_coarsening_32x4,
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
        flops=lambda m, k, n: 2 * m * k * n,
        nbytes=lambda m, k, n: 4 * (m * k + k * n + m * n),
        limits={"naive": 2048},
    )
