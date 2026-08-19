"""Vector add: the memory-bound baseline. 3 floats moved per 1 flop."""

from __future__ import annotations

import torch
from triton_dsl import vecadd_triton  # local package; not the `triton` wheel

from lab import harness

ext = harness.load_kernel(__file__)

VARIANTS = {
    "torch.add": torch.add,
    "cuda": ext.vecadd_cuda,
    "cuda_float4": ext.vecadd_cuda_float4,
    "triton": vecadd_triton,
}


def make_inputs(n: int):
    return (
        torch.randn(n, device="cuda", dtype=torch.float32),
        torch.randn(n, device="cuda", dtype=torch.float32),
    )


if __name__ == "__main__":
    harness.main(
        VARIANTS,
        make_inputs=make_inputs,
        shapes=[(1 << k,) for k in range(20, 26)],
        check_shapes=[(1,), (100,), (1024,), (1_000_003,)],
        ref=torch.add,
        # One add per element; read a and b, write c.
        flops=lambda n: n,
        nbytes=lambda n: 4 * 3 * n,
        tol={"rtol": 0.0, "atol": 0.0},  # fp32 add is exact, so demand exact
    )
