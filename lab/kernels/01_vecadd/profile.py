from __future__ import annotations

import matplotlib as mpl
import nsight
import torch
from reference import vecadd_ref

from lab.harness import RESULTS_DIR, load_kernel_extension

# nsight's savefig uses rcParams['savefig.dpi'] (default "figure" → 100).
mpl.rcParams["savefig.dpi"] = 200

module = load_kernel_extension("add_ext", __file__)
OUTPUT_DIR = RESULTS_DIR / "profiles" / "01_vecadd"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_IMAGE = OUTPUT_DIR / "vecadd_profile.png"

# Sweep vector lengths so each annotated kernel becomes a series vs input size.
SIZES = [(n,) for n in range(1 << 20, (16 << 20) + 1, 1 << 20)]
RUNS = 5


@nsight.analyze.plot(
    filename=str(PROFILE_IMAGE),
    title="Vector Add Execution Time",
    ylabel="gpu__time_duration.sum (avg: 5 runs)",
    plot_type="line",
    plot_width=12,
    plot_height=7,
)
@nsight.analyze.kernel(
    configs=SIZES,
    runs=RUNS,
    clock_control="base",
    output_prefix=str(OUTPUT_DIR / "vecadd_"),
    output_csv=True,
)
def profile_vecadd(n: int) -> None:
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)

    # Each annotated region launches exactly one kernel → one plot series.
    with nsight.annotate("torch.add"):
        vecadd_ref(a, b)
    with nsight.annotate("vecadd_cuda"):
        module.vecadd_cuda(a, b)
    with nsight.annotate("vecadd_cuda_float4"):
        module.vecadd_cuda_float4(a, b)


if __name__ == "__main__":
    results = profile_vecadd()
    frame = results.to_dataframe()
    print(frame.to_string(index=False))
    print(f"Profile image: {PROFILE_IMAGE}")
