from __future__ import annotations

from pathlib import Path
from typing import Iterable

from torch.utils.cpp_extension import load

from .paths import kernel_source_dir

DEFAULT_CUDA_FLAGS = ["-O3", "-lineinfo"]


def discover_sources(kernel_dir: str | Path, patterns: Iterable[str] = ("*.cpp", "*.cc", "*.cu")) -> list[Path]:
    root = Path(kernel_dir)
    sources: list[Path] = []
    for pattern in patterns:
        sources.extend(sorted(root.glob(pattern)))
    return sources


def _nvidia_wheel_paths() -> tuple[list[str], list[str]]:
    """Include/lib paths for pip-installed NVIDIA CUDA libraries (e.g. cuBLAS)."""
    try:
        import nvidia
    except ImportError:
        return [], []

    roots = [Path(path).resolve() for path in getattr(nvidia, "__path__", [])]
    include_dirs: list[str] = []
    lib_dirs: list[Path] = []
    for root in roots:
        # Prefer CUDA toolkit wheels (cu12/cu13), not cudnn/cusparselt.
        include_dirs.extend(
            str(path)
            for path in sorted(root.glob("cu[0-9]*/include"))
            if path.is_dir()
        )
        lib_dirs.extend(
            path for path in sorted(root.glob("cu[0-9]*/lib")) if path.is_dir()
        )

    ldflags: list[str] = []
    for lib_dir in lib_dirs:
        ldflags.append(f"-L{lib_dir}")
        ldflags.append(f"-Wl,-rpath,{lib_dir}")
        # Wheel ships versioned sonames (libcublas.so.13) without libcublas.so.
        for lib in sorted(lib_dir.glob("libcublas.so.*")):
            ldflags.append(f"-l:{lib.name}")
            break

    return include_dirs, ldflags


def load_extension(
    name: str,
    kernel_dir: str | Path,
    sources: Iterable[str | Path] | None = None,
    extra_cuda_cflags: Iterable[str] = DEFAULT_CUDA_FLAGS,
    extra_cflags: Iterable[str] | None = None,
    extra_ldflags: Iterable[str] | None = None,
    extra_include_paths: Iterable[str] | None = None,
    verbose: bool = True,
):
    """Compile and load a PyTorch C++/CUDA extension from a kernel directory.

    Expected directory shape:

    ```text
    ext.cpp
    kernel.cu
    test.py
    bench.py
    ```
    """
    root = Path(kernel_dir)
    source_paths = [Path(source) for source in sources] if sources is not None else discover_sources(root)
    if not source_paths:
        raise FileNotFoundError(f"No C++/CUDA sources found under {root}")

    nvidia_includes, nvidia_ldflags = _nvidia_wheel_paths()
    include_paths = list(extra_include_paths or []) + nvidia_includes
    ldflags = list(extra_ldflags or []) + nvidia_ldflags

    return load(
        name=name,
        sources=[str(path) for path in source_paths],
        extra_cuda_cflags=list(extra_cuda_cflags),
        extra_cflags=list(extra_cflags or []),
        extra_ldflags=ldflags,
        extra_include_paths=include_paths,
        verbose=verbose,
    )


def load_kernel_extension(
    name: str,
    file: str | Path,
    source_dir: str = "cuda",
    sources: Iterable[str | Path] | None = None,
    extra_cuda_cflags: Iterable[str] = DEFAULT_CUDA_FLAGS,
    verbose: bool = True,
):
    """Compile the CUDA/C++ sources next to a kernel script.

    A kernel folder usually looks like:

    ```text
    00_relu/
      README.md
      reference.py
      test.py
      bench.py
      cuda/
        ext.cpp
        relu.cu
    ```
    """
    return load_extension(
        name=name,
        kernel_dir=kernel_source_dir(file, source_dir),
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=verbose,
    )
