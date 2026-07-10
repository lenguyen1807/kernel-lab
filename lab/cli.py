from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

KERNELS_DIR = Path(__file__).resolve().parent / "kernels"


def ensure_profile_platform() -> None:
    if sys.platform != "linux":
        raise SystemExit(
            "cuda-lab profile must run on a CUDA-capable Linux host; "
            "macOS supports editing the profiling setup only."
        )


def run_kernel_script(kernel: str, script_name: str) -> None:
    kernel_dir = (KERNELS_DIR / kernel).resolve()
    script_path = kernel_dir / f"{script_name}.py"

    if not kernel_dir.is_dir():
        raise SystemExit(f"Unknown kernel folder: {kernel}")
    if not script_path.is_file():
        raise SystemExit(f"{kernel} has no {script_name}.py")

    sys.path.insert(0, str(kernel_dir))
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        try:
            sys.path.remove(str(kernel_dir))
        except ValueError:
            pass


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="cuda-lab",
        description="Run CUDA lab kernel tests, benchmarks, and profiles.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("test", "bench", "profile"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("kernel", help="kernel folder under lab/kernels")

    args = parser.parse_args(argv)
    if args.command == "profile":
        ensure_profile_platform()
    run_kernel_script(args.kernel, args.command)


if __name__ == "__main__":
    main()
