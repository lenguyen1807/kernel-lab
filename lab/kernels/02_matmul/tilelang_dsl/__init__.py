"""
- I want to test how fast tilelang is, just curious lol and it is fast af!
- The matmul kernel is copied from [General Matrix-Matrix Multiplication with Tile Library](https://tilelang.com/deeplearning_operators/matmul.html)
"""

import tilelang
import tilelang.language as T
import torch
from tilelang.cuda.intrinsics import make_mma_swizzle_layout

def matmul_kernel(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Optional layout hints (commented out by default)
            # T.annotate_layout({
            #     A_shared: make_mma_swizzle_layout(A_shared),
            #     B_shared: make_mma_swizzle_layout(B_shared),
            # })

            # Optional: Enabling swizzle-based rasterization
            # T.use_swizzle(panel_size=10, enable=True)

            # Clear local accumulation
            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # Copy tile of A from global to shared memory
                T.copy(A[by * block_M, ko * block_K], A_shared)

                # Parallel copy tile of B from global to shared memory
                for k, j in T.Parallel(block_K, block_N):
                    B_shared[k, j] = B[ko * block_K + k, bx * block_N + j]

                # Perform a tile-level GEMM
                T.gemm(A_shared, B_shared, C_local)

            # Copy result from local (register fragment) to global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main

# note that for tilelang, we need to JIT the kernel size first
def matmul_jit(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    func = matmul_kernel(M, N, K, block_M, block_N, block_K, dtype, accum_dtype)
    return tilelang.compile(func, out_idx=[2], target="cuda")

def matmul_factory(m: int, k: int, n: int):
    kernel = matmul_jit(m, n, k, 128, 128, 32)

    def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # the kernel is fp16; the lab's shared inputs are fp32
        return kernel(a.half(), b.half()).float()

    return run