from __future__ import annotations

import torch


def matmul_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch high-level matmul (often cuBLAS / cuBLASLt under the hood)."""
    return torch.matmul(a, b)
