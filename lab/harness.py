"""Compile, check, benchmark, and profile CUDA kernels.

One kernel folder = one `main.py`: load CUDA sources, declare a dict of variants,
hand both to `main()`, which provides three subcommands:

    test     correctness vs a reference          instant
    bench    latency sweep via CUDA events       seconds
    profile  Nsight Compute counters, one shape  minutes

`bench` answers *how fast*. `profile` answers *why*.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load

RESULTS = Path(__file__).resolve().parent.parent / "results"

# -lineinfo maps SASS back to source in Nsight; -Xptxas=-v prints per-kernel
# register and shared-memory usage, the cheapest occupancy signal there is.
DEFAULT_FLAGS = ("-O3", "-lineinfo", "-Xptxas=-v")

Shape = tuple[int, ...]


# --------------------------------------------------------------------- building


def _nvidia_wheel_paths() -> tuple[list[str], list[str]]:
    """Include/lib paths for pip-installed NVIDIA CUDA libraries (e.g. cuBLAS)."""
    try:
        import nvidia
    except ImportError:
        return [], []

    includes, ldflags = [], []
    for root in (Path(p).resolve() for p in getattr(nvidia, "__path__", [])):
        includes += [
            str(p) for p in sorted(root.glob("cu[0-9]*/include"))
        ]  # cu12/cu13, not cudnn
        for lib_dir in sorted(root.glob("cu[0-9]*/lib")):
            ldflags += [f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}"]
            # Wheel ships versioned sonames (libcublas.so.13), no libcublas.so.
            if libs := sorted(lib_dir.glob("libcublas.so.*")):
                ldflags.append(f"-l:{libs[0].name}")
    return includes, ldflags


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
    """Compile every C++/CUDA source under `<kernel folder>/cuda/` and load it.

    The extension name defaults to the kernel folder, so two folders never share
    a build cache (sharing one made torch rebuild on every switch).
    """
    kernel_dir = Path(file).resolve().parent
    sources = sorted(
        p
        for pat in ("*.cpp", "*.cc", "*.cu")
        for p in (kernel_dir / source_dir).glob(pat)
    )
    if not sources:
        raise FileNotFoundError(f"No C++/CUDA sources under {kernel_dir / source_dir}")
    includes, ldflags = _nvidia_wheel_paths()
    return load(
        name=name or f"{kernel_stem(kernel_dir)}_ext",
        sources=[str(p) for p in sources],
        extra_cuda_cflags=list(flags),
        extra_ldflags=ldflags,
        extra_include_paths=includes,
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


def _run_test(variants, make_inputs, shapes, ref, tol) -> None:
    if ref is None:
        raise SystemExit("test needs a reference: pass ref=... to harness.main()")

    for shape in shapes:
        kwargs = {"rtol": 1e-2, "atol": 1e-2}
        if tol:
            # tol may be a callable (*shape) -> mapping, for error bounds that
            # scale with the shape (e.g. fp16 rounding over a k-reduction).
            kwargs.update(tol(*shape) if callable(tol) else tol)
        args = make_inputs(*shape)
        expected = ref(*args)
        for label, fn in variants.items():
            if fn is ref:
                continue
            try:
                torch.testing.assert_close(
                    _resolve(fn, shape)(*args), expected, **kwargs
                )
            except AssertionError as exc:
                raise SystemExit(
                    f"\n{label} is wrong at {_fmt_shape(shape)}:\n{exc}"
                ) from None
        print(f"  ok  {_fmt_shape(shape)}")
    others = sum(fn is not ref for fn in variants.values())
    print(f"\n{others} variants match the reference across {len(shapes)} shapes.")


def _run_bench(
    variants, make_inputs, shapes, ref, flops, nbytes, limits, out_dir, plot
) -> None:
    print(f"GPU: {torch.cuda.get_device_name()}")
    ref_label = next((label for label, fn in variants.items() if fn is ref), None)

    headers = ["Kernel", "ms"]
    if flops:
        headers.append("TFLOP/s")
    if nbytes:
        headers.append("GB/s")
    if ref_label:
        headers.append(f"vs {ref_label}")

    timings: dict[Shape, dict[str, float | None]] = {}
    for shape in shapes:
        args = make_inputs(*shape)
        measured: dict[str, float | None] = {}
        for label, fn in variants.items():
            limit = limits.get(label)
            measured[label] = (
                None
                if limit is not None and max(shape) > limit
                else bench_ms(_resolve(fn, shape), *args)
            )
        timings[shape] = measured

        ref_ms = measured.get(ref_label)
        rows = []
        for label, ms in measured.items():
            tflops, gbs, vs = _bench_metrics(
                ms, shape, label, flops, nbytes, ref_label, ref_ms
            )
            if ms is None:
                rows.append([label, "skipped"] + [""] * (len(headers) - 2))
                continue
            row: list[object] = [label, f"{ms:.4f}"]
            if flops:
                row.append(f"{tflops:.2f}")
            if nbytes:
                row.append(f"{gbs:.1f}")
            if ref_label:
                row.append(f"{vs:.2f}x" if vs is not None else "")
            rows.append(row)
        print(f"\n### {_fmt_shape(shape)}\n\n{table(rows, headers)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "bench.csv"
    csv_headers = ["shape", "kernel", "latency_ms"]
    if flops:
        csv_headers.append("tflops")
    if nbytes:
        csv_headers.append("gb_s")
    if ref_label:
        csv_headers.append(f"vs_{ref_label}")
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
                if flops:
                    row.append("" if tflops is None else f"{tflops:.2f}")
                if nbytes:
                    row.append("" if gbs is None else f"{gbs:.1f}")
                if ref_label:
                    row.append("" if vs is None else f"{vs:.2f}")
                writer.writerow(row)
    print(f"\nCSV: {csv_path}")
    if plot:
        print(f"Plot: {_plot(timings, out_dir)}")


def _plot(timings: dict[Shape, dict[str, float | None]], out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(10, 6))
    for label in next(iter(timings.values())):
        points = [
            (max(s), timings[s][label])
            for s in timings
            if timings[s][label] is not None
        ]
        if points:
            axes.plot(
                [x for x, _ in points], [y for _, y in points], marker="o", label=label
            )
    axes.set(
        xscale="log", yscale="log", xlabel="size (largest dim)", ylabel="latency (ms)"
    )
    axes.set_title("Latency vs size (CUDA events, median)")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend()
    figure.tight_layout()
    path = out_dir / "bench.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _named_fn(func, name, params):
    """Rebuild `func` with explicit positional parameters `params`: nsight-python
    names artifacts after the function and config columns after its parameters,
    so a bare *args wrapper would produce unreadable CSVs."""
    namespace = {"_inner": func}
    exec(
        f"def {name}({', '.join(params)}): return _inner({', '.join(params)})",
        namespace,
    )
    return namespace[name]


def _nsight_sweep(
    variants, make_inputs, shapes, limits, out_dir, stem, runs, title, plot_path
):
    """Time the sweep with Nsight Compute instead of CUDA events. Costs minutes,
    not seconds: wall time scales with shapes x variants x runs."""
    import nsight

    print(f"GPU: {torch.cuda.get_device_name()} (Nsight locks clocks to base)")

    def skipped(label, shape):
        limit = limits.get(label)
        return limit is not None and max(shape) > limit

    dropped = sorted(
        {label for shape in shapes for label in variants if skipped(label, shape)}
    )
    if dropped:
        print(
            f"Skipping {', '.join(dropped)} above its limit -- use --shape to profile it smaller."
        )
    n = len(variants) - len(dropped)
    print(
        f"Nsight will replay ~{len(shapes) * n * runs} regions ({len(shapes)} shapes x {n} variants x {runs} runs)."
    )

    def body(*shape):
        args = make_inputs(*shape)
        for label, fn in variants.items():
            if not skipped(label, shape):
                with nsight.annotate(label):  # one region per launch -> one series
                    _resolve(fn, shape)(*args)

    body = _named_fn(
        body, f"{stem}_sweep", list(inspect.signature(make_inputs).parameters)
    )
    run = nsight.analyze.kernel(
        configs=[tuple(shape) for shape in shapes],
        runs=runs,
        clock_control="base",
        # torch/cuBLAS may launch several kernels per call; range replay sums them.
        replay_mode="range",
        combine_kernel_metrics=lambda x, y: x + y,
        output_prefix=str(out_dir / f"{stem}_"),
        output_csv=True,
    )(body)
    if plot_path:
        import matplotlib as mpl

        mpl.rcParams["savefig.dpi"] = (
            200  # nsight's savefig honours this, default is 100
        )
        run = nsight.analyze.plot(
            filename=str(plot_path),
            title=title,
            plot_type="line",
            ylabel=f"gpu__time_duration.sum (avg of {runs})",
            plot_width=12,
            plot_height=7,
        )(run)

    print(run().to_dataframe().to_string(index=False))
    print(f"\nArtifacts: {out_dir}")


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
) -> None:
    """Give a kernel folder its `test` / `bench` / `profile` commands.

    variants     {label: callable(*inputs) | ShapeJIT(factory)} -- ref too
    make_inputs  shape -> tuple of tensors, passed to every variant
    shapes       benchmark sweep; check_shapes defaults to it
    flops/nbytes shape -> count, for the TFLOP/s and GB/s columns
    limits       {label: max dim} -- skip a variant once it gets too slow
    tol          assert_close overrides (rtol/atol), or (*shape) -> mapping
    """
    # argv[0] is the kernel's main.py both via `cuda-lab` and standalone, so the
    # folder names the results dirs -- no load_kernel() call required.
    stem = kernel_stem(Path(sys.argv[0]).resolve().parent)
    shapes = [tuple(shape) for shape in shapes]
    limits = limits or {}

    parser = argparse.ArgumentParser(
        prog=f"cuda-lab ... {stem}", description=f"{stem} kernel lab"
    )
    parser.set_defaults(cmd="test")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("test", help="check every variant against the reference")
    bench = sub.add_parser("bench", help="latency sweep (seconds)")
    bench.add_argument("--plot", action="store_true", help="save a latency-vs-size PNG")
    bench.add_argument(
        "--nsight", action="store_true", help="re-time the sweep with Nsight (minutes)"
    )
    bench.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Nsight runs per point (default 1; locked clocks barely vary)",
    )
    profile = sub.add_parser("profile", help="Nsight counters at one shape (minutes)")
    profile.add_argument(
        "--shape",
        help=f"e.g. 2048 or 2048x2048x2048 (default {_fmt_shape(shapes[-1])})",
    )
    args = parser.parse_args()

    if args.cmd == "test":
        _run_test(variants, make_inputs, check_shapes or shapes, ref, tol)
    elif args.cmd == "bench" and args.nsight:
        out = RESULTS / "profiles" / stem
        _nsight_sweep(
            variants,
            make_inputs,
            shapes,
            limits,
            out,
            stem,
            args.runs,
            f"{stem}: execution time",
            out / f"{stem}_sweep.png",
        )
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
        )
    else:
        if args.shape:
            parts = [int(p) for p in re.split(r"[,x]", args.shape) if p]
            shape = tuple(parts * len(shapes[0]) if len(parts) == 1 else parts)
        else:
            shape = shapes[-1]
        print(f"Profiling {stem} at {_fmt_shape(shape)} (one shape, one run).")
        _nsight_sweep(
            variants,
            make_inputs,
            [shape],
            limits,
            RESULTS / "profiles" / stem,
            stem,
            1,
            f"{stem} @ {_fmt_shape(shape)}",
            None,
        )
