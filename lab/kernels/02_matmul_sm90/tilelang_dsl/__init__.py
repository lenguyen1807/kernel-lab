"""
Tilelang matmul under the folder contract: bf16 in/out, fp32 accumulate,
A row-major (M,K), B K-major -- declared here as an (N,K) row-major operand
so `b.t()` is a zero-copy view.

Adapted from the tilelang GEMM example:
https://tilelang.com/deeplearning_operators/matmul.html
"""

import tilelang
import tilelang.language as T
import torch


def matmul_kernel(M, N, K, block_M, block_N, block_K, dtype="bfloat16", accum_dtype="float"):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)

                # B is (N, K) row-major == the K-major (K, N) operand the
                # contract supplies; the parallel copy transposes on the fly.
                for k, j in T.Parallel(block_K, block_N):
                    B_shared[k, j] = B[bx * block_N + j, ko * block_K + k]

                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def matmul_jit(M, N, K, block_M, block_N, block_K, dtype="bfloat16", accum_dtype="float"):
    func = matmul_kernel(M, N, K, block_M, block_N, block_K, dtype, accum_dtype)
    return tilelang.compile(func, out_idx=[2], target="cuda")


def matmul_factory(m: int, k: int, n: int):
    kernel = matmul_jit(m, n, k, 128, 128, 32)

    def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # contract: b is (k, n) K-major; b.t() is the (n, k) row-major view
        # the kernel declares. No casts, no allocations.
        return kernel(a, b.t())

    return run
