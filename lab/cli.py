"""`cuda-lab <command> <kernel> [options]` -> `lab/kernels/<kernel>/main.py`.

Every kernel folder owns a single `main.py`; this only finds it and forwards
the command through. Running that script directly works just as well:

    python lab/kernels/02_matmul/main.py bench --plot
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

KERNELS_DIR = Path(__file__).resolve().parent / "kernels"
COMMANDS = ("test", "bench", "profile")


def available() -> list[str]:
    return sorted(
        path.parent.name
        for path in KERNELS_DIR.glob("*/main.py")
        if not path.parent.name.startswith(".")
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="cuda-lab",
        description="Test, benchmark, and profile CUDA kernels.",
        epilog="kernels: " + ", ".join(available()),
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("kernel", help="folder under lab/kernels")
    args, extra = parser.parse_known_args(argv)

    kernel_dir = (KERNELS_DIR / args.kernel).resolve()
    script = kernel_dir / "main.py"
    if not script.is_file():
        raise SystemExit(
            f"No lab/kernels/{args.kernel}/main.py. Available: {', '.join(available())}"
        )

    # Kernel folders may import sibling packages (e.g. 01_vecadd/triton_dsl).
    saved_argv = sys.argv
    sys.path.insert(0, str(kernel_dir))
    sys.argv = [str(script), args.command, *extra]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.path.remove(str(kernel_dir))
        sys.argv = saved_argv


if __name__ == "__main__":
    main()
