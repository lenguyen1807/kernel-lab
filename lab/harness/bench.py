from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import torch
from triton.testing import do_bench


@dataclass(frozen=True)
class BenchResult:
    name: str
    latency_ms: float
    tflops: float | None = None
    gflops: float | None = None
    bandwidth_gbs: float | None = None
    pct_peak: float | None = None
    notes: str = ""

    def as_row(self) -> list[str]:
        return [
            self.name,
            f"{self.latency_ms:.4f}",
            "" if self.tflops is None else f"{self.tflops:.2f}",
            "" if self.gflops is None else f"{self.gflops:.2f}",
            "" if self.bandwidth_gbs is None else f"{self.bandwidth_gbs:.2f}",
            "" if self.pct_peak is None else f"{self.pct_peak:.2f}",
            self.notes,
        ]


def benchmark_ms(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int = 25,
    rep: int = 100,
    **kwargs: Any,
) -> float:
    return float(
        do_bench(
            lambda: fn(*args, **kwargs), warmup=warmup, rep=rep, return_mode="median"
        )
    )


def torch_benchmark_ms(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int = 25,
    rep: int = 100,
    **kwargs: Any,
) -> float:
    assert torch.cuda.is_available(), "Must be run in CUDA environment"
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.median(samples)


def gbps(bytes_moved: float, latency_ms: float) -> float:
    return bytes_moved / latency_ms / 1e6


def gflops(flops: float, latency_ms: float) -> float:
    return flops / latency_ms / 1e6


def tflops(flops: float, latency_ms: float) -> float:
    return flops / latency_ms / 1e9


def pct_peak(value_tflops: float, peak_tflops: float | None) -> float | None:
    if not peak_tflops:
        return None
    return value_tflops / peak_tflops * 100.0


def format_markdown_table(
    rows: Iterable[BenchResult | Sequence[object]],
    headers: Sequence[str] = (
        "Kernel",
        "Latency (ms)",
        "TFLOPS",
        "GFLOPS",
        "GB/s",
        "% Peak",
        "Notes",
    ),
) -> str:
    normalized_rows: list[list[str]] = []
    for row in rows:
        if isinstance(row, BenchResult):
            normalized_rows.append(row.as_row())
        else:
            normalized_rows.append(
                ["" if value is None else str(value) for value in row]
            )

    widths = [len(header) for header in headers]
    for row in normalized_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def fmt(values: Sequence[str]) -> str:
        return (
            "| "
            + " | ".join(
                value.ljust(widths[index]) for index, value in enumerate(values)
            )
            + " |"
        )

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join(
        [fmt(list(headers)), separator, *(fmt(row) for row in normalized_rows)]
    )
