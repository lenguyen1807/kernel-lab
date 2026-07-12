from __future__ import annotations

import matplotlib as mpl
import nsight
import torch
from reference import matmul_ref

from lab.harness import RESULTS_DIR, load_kernel_extension

# nsight's savefig uses rcParams['savefig.dpi'] (default "figure" → 100).
mpl.rcParams["savefig.dpi"] = 200

module = load_kernel_extension("matmul_ext", __file__)
OUTPUT_DIR = RESULTS_DIR / "profiles" / "02_matmul"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_IMAGE = OUTPUT_DIR / "matmul_profile.png"

# Square N×N × N×N mats; coarsened kernels need N divisible by their tile sizes.
SIZES = [(n,) for n in range(256, 2049, 256)]
RUNS = 5


@nsight.analyze.plot(
    filename=str(PROFILE_IMAGE),
    title="Matmul Execution Time",
    ylabel="gpu__time_duration.sum (avg: 5 runs)",
    plot_type="line",
    plot_width=12,
    plot_height=7,
)
@nsight.analyze.kernel(
    configs=SIZES,
    runs=RUNS,
    clock_control="base",
    # torch/cuBLAS may launch multiple kernels per call; sum their durations.
    replay_mode="range",
    combine_kernel_metrics=lambda x, y: x + y,
    output_prefix=str(OUTPUT_DIR / "matmul_"),
    output_csv=True,
)
def profile_matmul(n: int) -> None:
    a = torch.randn(n, n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, n, device="cuda", dtype=torch.float32)

    # Each annotate label becomes one plot series.
    # torch.matmul may use cuBLAS/cuBLASLt/CUTLASS; cublas is explicit cublasSgemm.
    with nsight.annotate("torch"):
        matmul_ref(a, b)
    with nsight.annotate("cublas"):
        module.matmul_cublas(a, b)
    with nsight.annotate("naive"):
        module.matmul_naive(a, b)
    with nsight.annotate("tiled"):
        module.matmul_tiled(a, b)
    with nsight.annotate("1D_coarsening"):
        module.matmul_1D_coarsening(a, b)
    with nsight.annotate("2D_coarsening"):
        module.matmul_2D_coarsening(a, b)


if __name__ == "__main__":
    results = profile_matmul()
    frame = results.to_dataframe()
    print(frame.to_string(index=False))
    print(f"Profile image: {PROFILE_IMAGE}")
