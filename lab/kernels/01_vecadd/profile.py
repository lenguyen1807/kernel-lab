from __future__ import annotations

import nsight
import torch
from reference import vecadd_ref

from lab.harness import RESULTS_DIR, load_kernel_extension

module = load_kernel_extension("add_ext", __file__)
OUTPUT_DIR = RESULTS_DIR / "profiles" / "01_vecadd"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@nsight.analyze.kernel(
    configs=[16 * 1024 * 1024],
    runs=5,
    clock_control="base",
    output_prefix=str(OUTPUT_DIR / "vecadd_"),
    output_csv=True,
)
def profile_vecadd(n: int) -> None:
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)

    # Each annotated region launches exactly one kernel.
    with nsight.annotate("torch.add"):
        vecadd_ref(a, b)
    with nsight.annotate("vecadd_cuda"):
        module.vecadd_cuda(a, b)
    with nsight.annotate("vecadd_cuda_float4"):
        module.vecadd_cuda_float4(a, b)


if __name__ == "__main__":
    results = profile_vecadd()
    print(results.to_dataframe().to_string(index=False))
