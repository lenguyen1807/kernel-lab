from __future__ import annotations

import torch
from reference import matmul_ref

from lab.harness import assert_close, load_kernel_extension

module = load_kernel_extension("matmul_ext", __file__)

if __name__ == "__main__":
    for m, k, n in [(16, 16, 16), (31, 17, 29), (128, 256, 64), (1024, 1024, 1024)]:
        a = torch.randn(m, k, device="cuda", dtype=torch.float32)
        b = torch.randn(k, n, device="cuda", dtype=torch.float32)

        actual_cublas = module.matmul_cublas(a, b)
        actual = module.matmul_naive(a, b)
        actual_tiled = module.matmul_tiled(a, b)
        actual_1d_coarsening = module.matmul_1D_coarsening(a, b)
        actual_2d_coarsening = module.matmul_2D_coarsening(a, b)

        expected = matmul_ref(a, b)

        assert_close(actual_cublas, expected)
        assert_close(actual, expected)
        assert_close(actual_tiled, expected)
        assert_close(actual_1d_coarsening, expected)
        assert_close(actual_2d_coarsening, expected)

        print(f"Matmul is correct at shape ({m}, {k}) x ({k}, {n})")
