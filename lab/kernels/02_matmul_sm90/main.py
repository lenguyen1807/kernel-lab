from __future__ import annotations

import torch

from lab import harness
from tilelang_dsl import matmul_factory
from triton_dsl import matmul_triton_naive

ext = harness.load_kernel(__file__)

VARIANTS = {
    # bf16 torch.matmul -> cuBLAS HGEMM, fp32 accumulate.
    "torch": torch.matmul,
    "tilelang": harness.ShapeJIT(matmul_factory),
    "triton_naive": matmul_triton_naive,
    "2D_coarsening": ext.matmul_2D_coarsening
}


def make_inputs(m: int, k: int, n: int):
    scale = k**-0.5  # keep products ~O(1) so bf16 rounding stays honest
    return (
        torch.randn(m, k, device="cuda").mul(scale).bfloat16(),
        # logically (k, n), physically K-major
        torch.randn(n, k, device="cuda").mul(scale).bfloat16().T,
    )


if __name__ == "__main__":
    harness.main(
        VARIANTS,
        make_inputs=make_inputs,
        shapes=[(n, n, n) for n in (2048, 4096, 8192)],
        check_shapes=[(16, 16, 16), (31, 17, 29), (128, 256, 64), (1024, 1024, 1024)],
        ref=torch.matmul,
        flops=lambda m, k, n: 2 * m * k * n,
        nbytes=lambda m, k, n: 2 * (m * k + k * n + m * n),
        # dense bf16->fp32 tensor-core peaks. RTX PRO cards look full-rate
        # (4x fp32): cuBLAS already measured ~153 TFLOP/s, so the GeForce-style
        # 2x value (109.9) is provably wrong. Verify against your spec sheet.
        sol={"rtx pro 4500": 219.7, "a100": 312.0, "h100": 989.0, "5090": 209.5},
        num_input_sets=4,
        # bf16 in/out, fp32 accumulate: input rounding dominates, ~2^-8 rel
        tol={"rtol": 3e-2, "atol": 2e-2},
    )
