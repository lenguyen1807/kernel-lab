from __future__ import annotations

import torch
from reference import matmul_ref

from lab.harness import assert_close, load_kernel_extension

module = load_kernel_extension("matmul_ext", __file__)

if __name__ == "__main__":
    for m, k, n in [(16, 16, 16), (31, 17, 29), (128, 256, 64), (1024, 1024, 1024)]:
        a = torch.randn(m, k, device="cuda", dtype=torch.float32)
        b = torch.randn(k, n, device="cuda", dtype=torch.float32)

        actual = module.matmul_naive(a, b)
        actual_tiled_16 = module.matmul_tiled_16(a, b)
        actual_tiled_32 = module.matmul_tiled_32(a, b)
        actual_coarsening_16 = module.matmul_coarsening_16x4(a, b)
        actual_coarsening_32 = module.matmul_coarsening_32x4(a, b)

        expected = matmul_ref(a, b)

        assert_close(actual, expected)
        assert_close(actual_tiled_16, expected)
        assert_close(actual_tiled_32, expected)
        assert_close(actual_coarsening_16, expected)
        assert_close(actual_coarsening_32, expected)

        print(f"Matmul is correct at shape ({m}, {k}) x ({k}, {n})")
