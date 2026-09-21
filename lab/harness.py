"""Compile, check, benchmark, and profile GPU kernels (CUDA; ROCm-aware).

One kernel folder = one `main.py`: load CUDA sources, declare a dict of variants,
hand both to `main()`, which provides three subcommands:

    test     correctness vs a reference          instant
    bench    latency + peak-memory sweep         seconds
    profile  .ncu-rep reports, one shape         minutes

`bench` answers *how fast*. `profile` answers *why*.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load

RESULTS = Path(__file__).resolve().parent.parent / "results"

# -lineinfo maps SASS back to source in Nsight; -Xptxas=-v prints per-kernel
# register and shared-memory usage, the cheapest occupancy signal there is.
DEFAULT_FLAGS = ("-O3", "-lineinfo", "-Xptxas=-v")
# hipcc is clang: no -Xptxas, and -lineinfo is nvcc spelling.
HIP_FLAGS = ("-O3",)

Shape = tuple[int, ...]


def _is_hip() -> bool:
    return torch.version.hip is not None


def _gpu_tag() -> str:
    """Filename-safe GPU id: 'NVIDIA H100 SXM5' -> 'h100_sxm5'. Every artifact
    the harness writes is tagged so results from different machines coexist."""
    name = torch.cuda.get_device_name().lower()
    for word in ("nvidia", "geforce", "amd", "radeon", "instinct", "corporation"):
        name = name.replace(word, "")
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_") or "gpu"


# --------------------------------------------------------------------- building


def _cublas_flags() -> tuple[list[str], list[str]]:
    """(extra nvcc flags, extra ldflags) for cuBLAS. Headers and libs come from
    the CUDA_HOME toolkit when it ships cuBLAS -- CCCL hard-errors when the
    compiler and toolkit headers come from different releases, so the wheel
    include dir must never shadow them. The pip NVIDIA wheels are only a
    fallback for toolkits without cuBLAS, include dir trailing as -isystem."""
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME and (Path(CUDA_HOME) / "include" / "cublas_v2.h").exists():
        return [], ["-lcublas"]

    import nvidia

    cuda_flags, ldflags = [], []
    for root in (Path(p).resolve() for p in getattr(nvidia, "__path__", [])):
        for include_dir in sorted(root.glob("cu[0-9]*/include")):  # cu12/cu13, not cudnn
            cuda_flags += ["-isystem", str(include_dir)]
        for lib_dir in sorted(root.glob("cu[0-9]*/lib")):
            ldflags += [f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}"]
            # Wheel ships versioned sonames (libcublas.so.13), no libcublas.so.
            if libs := sorted(lib_dir.glob("libcublas.so.*")):
                ldflags.append(f"-l:{libs[0].name}")
    return cuda_flags, ldflags


def kernel_stem(kernel_dir: Path) -> str:
    """`02_matmul_pmpp` -> `matmul_pmpp`. Strips the ordering prefix."""
    return re.sub(r"^\d+[a-z]?_?", "", kernel_dir.name) or kernel_dir.name


def load_kernel(
    file: str | Path,
    *,
    name: str | None = None,
    source_dir: str = "cuda",
    flags: Sequence[str] = DEFAULT_FLAGS,
    verbose: bool = True,
):
    """Compile every C++/CUDA/HIP source under `<kernel folder>/<source_dir>/`.

    NVIDIA builds come from `cuda/` (nvcc), AMD builds from `hip/` (native HIP
    via hipcc -- torch recognizes .hip as a GPU source only on ROCm builds and
    passes it through without hipifying). The extension name defaults to the
    kernel folder, so two folders never share a build cache (sharing one made
    torch rebuild on every switch).
    """
    kernel_dir = Path(file).resolve().parent
    sources = sorted(
        p
        for pat in ("*.cpp", "*.cc", "*.cu", "*.hip")
        for p in (kernel_dir / source_dir).glob(pat)
    )
    if not sources:
        raise FileNotFoundError(
            f"No C++/CUDA/HIP sources under {kernel_dir / source_dir}"
        )
    if _is_hip():
        # Native HIP path: hipcc is clang, so nvcc flags and the NVIDIA cuBLAS
        # paths don't apply. (A .cu here would be hipified first -- portable,
        # but hip/ exists precisely to write AMD-native source instead.)
        extra_flags, extra_ldflags = [], []
        flags = HIP_FLAGS if tuple(flags) == DEFAULT_FLAGS else flags
    else:
        extra_flags, extra_ldflags = _cublas_flags()
    return load(
        name=name or f"{kernel_stem(kernel_dir)}_ext",
        sources=[str(p) for p in sources],
        extra_cuda_cflags=[*flags, *extra_flags],
        extra_ldflags=extra_ldflags,
        verbose=verbose,
    )


class ShapeJIT:
    """A variant that must be compiled per problem shape (tilelang, AoT tuning).

    Wrap a factory `(*shape) -> callable(*inputs)`; the harness calls it once
    per shape, caches the result, and treats the compiled callable like any
    other variant -- so JIT cost lands outside the timed region.
    """

    def __init__(self, factory: Callable[..., Callable]) -> None:
        self._factory = factory
        self._cache: dict[Shape, Callable] = {}

    def for_shape(self, shape: Shape) -> Callable:
        if shape not in self._cache:
            self._cache[shape] = self._factory(*shape)
        return self._cache[shape]


def _resolve(fn: Callable, shape: Shape) -> Callable:
    return fn.for_shape(shape) if isinstance(fn, ShapeJIT) else fn


# ---------------------------------------------------------------------- timing


def bench_ms(
    fn: Callable[..., Any],
    *args: Any,
    warmup_ms: int = 25,
    rep_ms: int = 100,
    **kwargs: Any,
) -> float:
    """Median latency in ms. warmup_ms/rep_ms are time budgets: do_bench picks an
    iteration count that fills the budget and clears L2 between iterations."""
    from triton.testing import do_bench

    return float(
        do_bench(
            lambda: fn(*args, **kwargs),
            warmup=warmup_ms,
            rep=rep_ms,
            return_mode="median",
        )
    )


def _cycling(fn: Callable[..., Any], inputs: Iterator) -> Callable[[], Any]:
    """Zero-arg timed call rotating through the input sets. Binds `fn` and
    `inputs` as parameters, so the closure never depends on loop state."""
    return lambda: fn(*next(inputs))


def peak_alloc_bytes(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> int:
    """Extra device memory (bytes) one call allocates above its live inputs --
    the S/P materialization a fused kernel exists to erase. Counts live
    tensors, not the caching allocator's reserved size."""
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - before


def _alloc_retries() -> int:
    """Allocator near-misses so far: cudaMallocs that failed, evicted the
    cached blocks, and were retried (the W921 stderr warnings). The retried
    allocation usually still succeeds -- this counts pressure, not failure.
    Read only after CUDA is initialized, or the keys are absent."""
    return int(torch.cuda.memory_stats().get("num_alloc_retries", 0))


_SDPA_OPS = {
    "_scaled_dot_product_flash_attention": "flash",
    "_scaled_dot_product_efficient_attention": "mem_efficient",
    "_scaled_dot_product_cudnn_attention": "cudnn",
}


def sdpa_backend_used(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str | None:
    """Which fused SDPA backend actually serviced one call of `fn`.

    `torch.sdpa` is a dispatcher: the winner depends on rank, dtype, head_dim,
    and mask, and nothing in the output betrays the choice. Two stacked spy
    modes settle it without a profiler (kineto/CUPTI init prints USDT noise
    on every session): __torch_function__ sees the public sdpa entry for any
    backend, __torch_dispatch__ sees the fused op the dispatcher really ran.
    "math" means the composite fallback (bmm + softmax) ran instead of a
    fused kernel; None means the call never touched sdpa at all.
    """
    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import TorchDispatchMode

    called: list[bool] = []
    fused: list[str] = []

    class FnSpy(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            if "scaled_dot_product_attention" in str(func):
                called.append(True)
            return func(*args, **(kwargs or {}))

    class DispatchSpy(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = str(func)
            for op, backend in _SDPA_OPS.items():
                if op in name:
                    fused.append(backend)
            return func(*args, **(kwargs or {}))

    with FnSpy(), DispatchSpy():
        fn(*args, **kwargs)
    if fused:
        return fused[0]
    return "math" if called else None


# ---------------------------------------------------------------------- output


def table(rows: Sequence[Sequence[object]], headers: Sequence[str]) -> str:
    """Markdown table; first column left-aligned, the rest right-aligned."""
    cells = [["" if v is None else str(v) for v in row] for row in rows]
    widths = [
        max([len(str(h)), *(len(r[i]) for r in cells)]) for i, h in enumerate(headers)
    ]

    def line(values: Sequence[object]) -> str:
        vals = [str(v) for v in values]
        return (
            "| "
            + " | ".join(
                v.ljust(widths[i]) if i == 0 else v.rjust(widths[i])
                for i, v in enumerate(vals)
            )
            + " |"
        )

    rule = (
        "| "
        + " | ".join(
            "-" * w if i == 0 else "-" * (w - 1) + ":" for i, w in enumerate(widths)
        )
        + " |"
    )
    return "\n".join([line(headers), rule, *(line(r) for r in cells)])


def _fmt_shape(shape: Shape) -> str:
    return "x".join(map(str, shape))


def _fmt_bytes(n: float) -> str:
    """1234567 -> '1.2 MiB', so peak-memory cells stay compact at any scale."""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.0f} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def _bench_metrics(ms, shape, label, flops, nbytes, ref_label, ref_ms):
    """TFLOP/s, GB/s, and speedup vs the reference. None when skipped or unused."""
    if ms is None:
        return None, None, None
    tflops = flops(*shape) / ms / 1e9 if flops else None
    gbs = nbytes(*shape) / ms / 1e6 if nbytes else None
    if not ref_label:
        vs = None
    elif label == ref_label:
        vs = 1.0
    elif ref_ms:
        vs = ref_ms / ms
    else:
        vs = None
    return tflops, gbs, vs


# ------------------------------------------------------------------ subcommands


def _tol_kwargs(tol, shape) -> dict:
    """assert_close tolerances for one shape. tol may be a callable (*shape)
    -> mapping, for error bounds that scale with the shape (e.g. bf16 input
    rounding accumulated over a k-reduction)."""
    kwargs = {"rtol": 1e-2, "atol": 1e-2}
    if tol:
        kwargs.update(tol(*shape) if callable(tol) else tol)
    return kwargs


def _resolve_sol(sol) -> float | None:
    """Peak TFLOP/s for the folder's dtype on the current GPU. `sol` is a
    scalar or a {name-substring: tflops} map so one main.py can serve several
    machines; a miss just drops the %SOL column."""
    if sol is None:
        return None
    if isinstance(sol, Mapping):
        name = torch.cuda.get_device_name().lower()
        for key, tflops in sol.items():
            if key.lower() in name:
                return float(tflops)
        print(f"note: no SOL peak declared for {name!r}; %SOL column skipped.")
        return None
    return float(sol)


def _check_variant(label, fn, shape, args, expected, tol) -> None:
    try:
        torch.testing.assert_close(
            _resolve(fn, shape)(*args), expected, **_tol_kwargs(tol, shape)
        )
    except AssertionError as exc:
        raise SystemExit(f"\n{label} is wrong at {_fmt_shape(shape)}:\n{exc}") from None


def _run_test(variants, make_inputs, shapes, ref, tol) -> None:
    if ref is None:
        raise SystemExit("test needs a reference: pass ref=... to harness.main()")

    for shape in shapes:
        args = make_inputs(*shape)
        expected = ref(*args)
        for label, fn in variants.items():
            if fn is ref:
                continue
            _check_variant(label, fn, shape, args, expected, tol)
        print(f"  ok  {_fmt_shape(shape)}")
    others = sum(fn is not ref for fn in variants.values())
    print(f"\n{others} variants match the reference across {len(shapes)} shapes.")


def _run_bench(
    variants, make_inputs, shapes, ref, flops, nbytes, limits, out_dir, plot,
    tol=None, sol=None, num_input_sets=1, check=True,
) -> None:
    """Sweep latency (do_bench median), peak allocation, and the sdpa
    backend actually used, one table per shape. A variant that OOMs is
    recorded and skipped at larger shapes; allocator near-misses (retried
    cudaMallocs) are starred and counted. The sweep survives both."""
    print(
        f"GPU: {torch.cuda.get_device_name()} | torch {torch.__version__}"
        f" | CUDA {torch.version.cuda}"
    )
    ref_label = next((label for label, fn in variants.items() if fn is ref), None)
    sol_tflops = _resolve_sol(sol)

    headers = ["Kernel", "ms", "peak mem"]
    if flops:
        headers.append("TFLOP/s")
        if sol_tflops:
            headers.append("%SOL")
    if nbytes:
        headers.append("GB/s")
    if ref_label:
        headers.append(f"vs {ref_label}")
    headers.append("sdpa")

    timings: dict[Shape, dict[str, float | None]] = {}
    mems: dict[Shape, dict[str, int | None]] = {}
    backends: dict[Shape, dict[str, str | None]] = {}
    statuses: dict[Shape, dict[str, str]] = {}
    retries: dict[Shape, dict[str, int]] = {}
    # A variant that OOMs at one shape is hopeless above it: remember the
    # shape, skip the larger ones, and let the sweep survive.
    oom_at: dict[str, Shape] = {}
    for shape in shapes:
        # Distinct input sets cycled across reps: on top of do_bench's L2
        # flush, nothing is L2-resident or bitwise-identical between reps.
        args_list = [make_inputs(*shape) for _ in range(num_input_sets)]

        measured: dict[str, float | None] = {}
        shape_mems: dict[str, int | None] = {}
        shape_backends: dict[str, str | None] = {}
        status: dict[str, str] = {}
        shape_retries: dict[str, int] = {label: 0 for label in variants}

        # Gate: never time a kernel that computes garbage.
        if check and ref is not None:
            try:
                expected = ref(*args_list[0])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"\n  ref {ref_label} OOM at {_fmt_shape(shape)}; shape skipped.")
                for label in variants:
                    measured[label] = None
                    shape_mems[label] = None
                    shape_backends[label] = None
                    status[label] = "ref_oom"
                timings[shape] = measured
                mems[shape] = shape_mems
                backends[shape] = shape_backends
                statuses[shape] = status
                continue
            for label, fn in variants.items():
                if fn is ref or _over_limit(limits, label, shape) or label in oom_at:
                    continue
                retries_before = _alloc_retries()
                try:
                    _check_variant(label, fn, shape, args_list[0], expected, tol)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    oom_at[label] = shape
                    print(f"  {label} OOM at {_fmt_shape(shape)} during check.")
                shape_retries[label] += _alloc_retries() - retries_before

        for label, fn in variants.items():
            if label in oom_at:
                status[label] = "oom" if oom_at[label] == shape else "skipped_oom"
                measured[label] = None
                shape_mems[label] = None
                shape_backends[label] = None
                continue
            if _over_limit(limits, label, shape):
                status[label] = "skipped"
                measured[label] = None
                shape_mems[label] = None
                shape_backends[label] = None
                continue
            status[label] = "ok"
            resolved = _resolve(fn, shape)
            inputs = iter(itertools.cycle(args_list))
            retries_before = _alloc_retries()
            try:
                measured[label] = bench_ms(_cycling(resolved, inputs))
                # One untimed call each, after the timed loop: allocator warm,
                # JIT/autotune spent. Both measurements are value-independent,
                # so args_list[0] is as good as any rep's inputs.
                shape_mems[label] = peak_alloc_bytes(resolved, *args_list[0])
                shape_backends[label] = sdpa_backend_used(resolved, *args_list[0])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                oom_at[label] = shape
                status[label] = "oom"
                measured[label] = None
                shape_mems[label] = None
                shape_backends[label] = None
                print(f"  {label} OOM at {_fmt_shape(shape)}; larger shapes skip it.")
            shape_retries[label] += _alloc_retries() - retries_before
        timings[shape] = measured
        mems[shape] = shape_mems
        backends[shape] = shape_backends
        statuses[shape] = status
        retries[shape] = shape_retries

        ref_ms = measured.get(ref_label)
        rows = []
        for label, ms in measured.items():
            tflops, gbs, vs = _bench_metrics(
                ms, shape, label, flops, nbytes, ref_label, ref_ms
            )
            if ms is None:
                why = {
                    "skipped": "skipped",
                    "oom": "OOM",
                    "skipped_oom": "skip>OOM",
                    "ref_oom": "ref OOM",
                }[status[label]]
                rows.append([label, why] + [""] * (len(headers) - 2))
                continue
            cell = f"{ms:.4f}" + ("*" if shape_retries[label] else "")
            row: list[object] = [label, cell, _fmt_bytes(shape_mems[label])]
            if flops:
                row.append(f"{tflops:.2f}")
                if sol_tflops:
                    row.append(f"{tflops / sol_tflops:.1%}")
            if nbytes:
                row.append(f"{gbs:.1f}")
            if ref_label:
                row.append(f"{vs:.2f}x" if vs is not None else "")
            row.append(shape_backends[label] or "-")
            rows.append(row)
        print(f"\n### {_fmt_shape(shape)}\n\n{table(rows, headers)}")
        near = {l: r for l, r in shape_retries.items() if r}
        if near:
            detail = ", ".join(f"{l} x{r}" for l, r in near.items())
            print(
                f"\n*: allocator near-miss -- cudaMalloc retried after evicting "
                f"cached blocks (W921 on stderr); the allocation still "
                f"succeeded. {detail}"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _gpu_tag()
    csv_path = out_dir / f"bench_{tag}.csv"
    csv_headers = ["shape", "kernel", "latency_ms", "peak_mem_bytes"]
    if flops:
        csv_headers.append("tflops")
        if sol_tflops:
            csv_headers.append("pct_sol")
    if nbytes:
        csv_headers.append("gb_s")
    if ref_label:
        csv_headers.append(f"vs_{ref_label}")
    csv_headers.append("sdpa_backend")
    csv_headers.append("alloc_retries")
    csv_headers.append("status")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(csv_headers)
        for shape, measured in timings.items():
            ref_ms = measured.get(ref_label)
            for label, ms in measured.items():
                tflops, gbs, vs = _bench_metrics(
                    ms, shape, label, flops, nbytes, ref_label, ref_ms
                )
                row = [_fmt_shape(shape), label, "" if ms is None else f"{ms:.6f}"]
                mem = mems[shape][label]
                row.append("" if mem is None else str(mem))
                if flops:
                    row.append("" if tflops is None else f"{tflops:.2f}")
                    if sol_tflops:
                        row.append(
                            "" if tflops is None else f"{tflops / sol_tflops:.4f}"
                        )
                if nbytes:
                    row.append("" if gbs is None else f"{gbs:.1f}")
                if ref_label:
                    row.append("" if vs is None else f"{vs:.2f}")
                row.append(backends[shape][label] or "")
                row.append(retries[shape][label])
                row.append(statuses[shape][label])
                writer.writerow(row)
    for label, shape in oom_at.items():
        print(f"OOM recorded: {label} at {_fmt_shape(shape)} (skipped above it)")
    print(f"\nCSV: {csv_path}")
    if plot:
        path = out_dir / f"bench_{tag}.png"
        print(f"Plot: {_plot(timings, path, 'Latency vs size (CUDA events, median)')}")
        mem_path = out_dir / f"bench_{tag}_mem.png"
        plotted = _plot(
            mems,
            mem_path,
            "Peak allocated memory vs size",
            ylabel="bytes above live inputs",
        )
        print(f"Plot: {plotted}")


def _plot(
    series: dict[Shape, dict[str, float | None]],
    path: Path,
    title: str,
    ylabel: str = "latency (ms)",
) -> Path:
    """Log-log curve of one metric per variant; skipped points drop out."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(10, 6))
    for label in next(iter(series.values())):
        points = [
            (max(s), series[s][label])
            for s in series
            if series[s][label] is not None
        ]
        if points:
            axes.plot(
                [x for x, _ in points], [y for _, y in points], marker="o", label=label
            )
    axes.set(
        xscale="log", yscale="log", xlabel="size (largest dim)", ylabel=ylabel
    )
    axes.set_title(title)
    axes.grid(True, which="both", alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _over_limit(limits, label, shape) -> bool:
    limit = limits.get(label)
    return limit is not None and max(shape) > limit


def _profile_ncu_rep(variants, shape, limits, out_dir, stem) -> None:
    """One .ncu-rep per variant at a single shape, for interactive inspection
    in the Nsight Compute GUI. Always drives the ncu CLI directly, filtered to
    a single NVTX-annotated launch via the `_nvtx-run` probe."""
    if _is_hip():
        raise SystemExit(
            "profile is Nsight-only so far; the rocprof backend lands with the "
            "ROCm track."
        )
    ncu = shutil.which("ncu")
    if ncu is None:
        raise SystemExit("Cannot profile: no ncu on PATH.")
    print(f"GPU: {torch.cuda.get_device_name()} (Nsight locks clocks to base)")

    script = Path(sys.argv[0]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    shape_str = _fmt_shape(shape)
    for label in variants:
        if _over_limit(limits, label, shape):
            print(f"  skip {label:<12} above its limit -- use --shape to go smaller.")
            continue
        report = out_dir / f"{stem}_{label}_{shape_str}"
        cmd = [
            ncu,
            "--nvtx",
            "--nvtx-include",
            f"{label}/",
            "--clock-control",
            "base",
            "--set",
            "full",
            "-f",
            "-o",
            str(report),
            sys.executable,
            str(script),
            "_nvtx-run",
            label,
            shape_str,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            # ncu prints its ==ERROR== diagnostics to stdout, not stderr.
            sys.stderr.write(proc.stdout + proc.stderr)
            raise SystemExit(f"ncu failed for {label} @ {shape_str} (exit {proc.returncode})")
        print(f"  {label:<12} -> {report}.ncu-rep")
    print(f"\nOpen with: ncu-ui {out_dir}/<file>.ncu-rep")


# ------------------------------------------------------------------ profile


def _parse_shape(text: str, dims: int) -> Shape:
    """'2048' or '2048x1024x512' -> tuple; a bare number broadcasts to `dims`."""
    parts = [int(p) for p in re.split(r"[,x]", text) if p]
    return tuple(parts * dims if len(parts) == 1 else parts)


def _nvtx_probe(variants, make_inputs, label, shape_str, dims, stem) -> None:
    """Run one variant once inside nested NVTX ranges `<stem>` > `<label>`, so
    ncu can profile exactly that launch; `--nvtx-include` matches any enclosing
    range, so both names work as filters."""
    if label not in variants:
        raise SystemExit(f"unknown variant {label!r}; choices: {', '.join(variants)}")
    shape = _parse_shape(shape_str, dims)
    fn = _resolve(variants[label], shape)
    args = make_inputs(*shape)
    fn(*args)  # warm up JIT/autotune outside the profiled range
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(stem)
    torch.cuda.nvtx.range_push(label)
    fn(*args)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print(f"profiled {label} @ {_fmt_shape(shape)} (NVTX ranges: {stem}/{label})", file=sys.stderr)


# ----------------------------------------------------------------------- entry


def main(
    variants: Mapping[str, Callable],
    *,
    make_inputs: Callable[..., tuple],
    shapes: Sequence[Shape],
    check_shapes: Sequence[Shape] | None = None,
    ref: Callable | None = None,
    flops: Callable[..., float] | None = None,
    nbytes: Callable[..., float] | None = None,
    limits: Mapping[str, int] | None = None,
    tol: Mapping[str, float] | Callable[..., Mapping[str, float]] | None = None,
    sol: float | Mapping[str, float] | None = None,
    num_input_sets: int = 1,
    check: bool = True,
) -> None:
    """Give a kernel folder its `test` / `bench` / `profile` commands.

    variants     {label: callable(*inputs) | ShapeJIT(factory)} -- ref too
    make_inputs  shape -> tuple of tensors, passed to every variant
    shapes       benchmark sweep; check_shapes defaults to it
    flops/nbytes shape -> count, for the TFLOP/s and GB/s columns
    limits       {label: max dim} -- skip a variant once it gets too slow
    tol          assert_close overrides (rtol/atol), or (*shape) -> mapping
    sol          peak TFLOP/s for the folder's dtype, scalar or
                 {gpu-name-substring: tflops}; enables a %SOL column
    num_input_sets distinct input sets cycled across timed reps (>1 is
                 stricter: nothing is bitwise-identical between reps)
    check        verify each variant vs ref before timing it
    """
    # argv[0] is the kernel's main.py both via `cuda-lab` and standalone, so the
    # folder names the results dirs -- no load_kernel() call required.
    stem = kernel_stem(Path(sys.argv[0]).resolve().parent)
    shapes = [tuple(shape) for shape in shapes]
    limits = limits or {}

    # Hidden entry point for ncu profiling, invoked as
    # `python main.py _nvtx-run <label> <shape>`; not part of the public CLI.
    if len(sys.argv) > 1 and sys.argv[1] == "_nvtx-run":
        _nvtx_probe(
            variants, make_inputs, sys.argv[2], sys.argv[3], len(shapes[0]), stem
        )
        return

    parser = argparse.ArgumentParser(
        prog=f"kernel-lab ... {stem}", description=f"{stem} kernel lab"
    )
    parser.set_defaults(cmd="test")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("test", help="check every variant against the reference")
    bench = sub.add_parser("bench", help="latency + peak-memory sweep (seconds)")
    bench.add_argument(
        "--plot",
        action="store_true",
        help="save latency- and peak-memory-vs-size PNGs",
    )
    profile = sub.add_parser(
        "profile", help="write one .ncu-rep per variant at a single shape (minutes)"
    )
    profile.add_argument(
        "--shape",
        help=f"e.g. 2048 or 2048x2048x2048 (default {_fmt_shape(shapes[-1])})",
    )
    args = parser.parse_args()

    if args.cmd == "test":
        _run_test(variants, make_inputs, check_shapes or shapes, ref, tol)
    elif args.cmd == "bench":
        _run_bench(
            variants,
            make_inputs,
            shapes,
            ref,
            flops,
            nbytes,
            limits,
            RESULTS / "bench" / stem,
            args.plot,
            tol=tol,
            sol=sol,
            num_input_sets=num_input_sets,
            check=check,
        )
    else:
        shape = _parse_shape(args.shape, len(shapes[0])) if args.shape else shapes[-1]
        print(f"Profiling {stem} at {_fmt_shape(shape)} (one shape, one launch per variant).")
        _profile_ncu_rep(variants, shape, limits, RESULTS / "profiles" / stem, stem)
