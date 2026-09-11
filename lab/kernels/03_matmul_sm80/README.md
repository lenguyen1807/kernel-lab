# Matmul — tensor cores (sm80-style `mma.sync`, runs sm80 → sm120)

**Folder contract:** bf16 in/out, fp32 accumulate. A is (M,K) row-major, B is
(K,N) K-major (TN GEMM — the operand layout `mma`/`ldmatrix` and cuBLAS-TN
both want). A variant is `f(A, B) -> C` with no casts or allocations inside
the timed call. Inputs are scaled by K^-0.5.

Why a separate folder: `02_matmul` is the fp32 SIMT ladder — comparing it
against a tensor-core kernel measures hardware peaks, not kernel quality. The
tensor-core ISA differs per arch: this folder targets `mma.sync m16n8k16` +
`ldmatrix` + `cp.async`, which runs on sm80 (A100) through sm120 (RTX PRO
4500, RTX 5090). tcgen05/tmem (sm100, B200) and wgmma (sm90, H100) are
different programs and get their own folders when the hardware exists.

Reference: `02_matmul_sm80/` and `02_matmul_sm120/` in
[gau-nernst/learn-cuda](https://github.com/gau-nernst/learn-cuda).

## Problem and baseline

cuBLAS HGEMM via `torch.matmul` (bf16). On the RTX PRO 4500 the dense
bf16→fp32 peak is ~110 TFLOP/s (2× fp32 — consumer-rate tensor cores), so the
%SOL column is the honest yardstick.

## Variants and expected bottleneck

## FLOPs, bytes, arithmetic intensity

## Roofline expectation

## Measured

## What failed, and what's next
