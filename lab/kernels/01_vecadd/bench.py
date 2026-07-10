from __future__ import annotations

import torch
from reference import vecadd_ref

from lab.harness import (
    BenchResult,
    RooflineEstimate,
    benchmark_ms,
    format_markdown_table,
    load_kernel_extension,
    total_bytes,
)

module = load_kernel_extension("add_ext", __file__)


if __name__ == "__main__":
    n = 16 * 1024 * 1024
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)

    estimate = RooflineEstimate(
        name="vecadd_fp32",
        # read a, b twice and write to c once
        # so total read is 2 * n and write is n
        bytes_moved=total_bytes(a.element_size(), reads=2 * n, writes=n),
        flops=n,  # one add per output element
    )

    torch_latency = benchmark_ms(vecadd_ref, a, b)
    cuda_latency = benchmark_ms(module.vecadd_cuda, a, b)
    cuda_float4_latency = benchmark_ms(module.vecadd_cuda_float4, a, b)

    results = [
        BenchResult(
            "torch.add",
            torch_latency,
            bandwidth_gbs=estimate.achieved_gbps(torch_latency),
            gflops=estimate.achieved_gflops(torch_latency),
            notes="Pytorch baseline",
        ),
        BenchResult(
            "vecadd_cuda",
            cuda_latency,
            bandwidth_gbs=estimate.achieved_gbps(cuda_latency),
            gflops=estimate.achieved_gflops(cuda_latency),
            notes="Vector add naive",
        ),
        BenchResult(
            "vecadd_cuda_float4",
            cuda_float4_latency,
            bandwidth_gbs=estimate.achieved_gbps(cuda_float4_latency),
            gflops=estimate.achieved_gflops(cuda_float4_latency),
            notes="Vector add with float4-vectorized",
        ),
    ]

    print(format_markdown_table(results))
