from __future__ import annotations

import torch
import torch.nn.functional as F
from lab import harness
from naive import attention_naive
from torch.nn.attention import SDPBackend, sdpa_kernel


@sdpa_kernel([SDPBackend.FLASH_ATTENTION])
def torch_baseline(q, k, v, causal_mask: bool = False):
    # note that the kernel accept 4D Q, K, V
    # [B, H, N, d]
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal_mask)


VARIANTS = {"torch.sdpa": torch_baseline, "naive": attention_naive}


def make_inputs(seq_len: int, head: int, hidden_dim: int, causal_mask: bool = False):
    scale = hidden_dim**-0.5
    return (
        torch.randn(1, head, seq_len, hidden_dim, device="cuda").mul(scale).bfloat16(),
        torch.randn(1, head, seq_len, hidden_dim, device="cuda").mul(scale).bfloat16(),
        torch.randn(1, head, seq_len, hidden_dim, device="cuda").mul(scale).bfloat16(),
        causal_mask
    )


if __name__ == "__main__":
    harness.main(
        VARIANTS,
        make_inputs=make_inputs,
        # we assume batch = 1, h = 16, d = 128
        shapes=[(n, 16, 128) for n in (4096, 8192, 16384, 32656)],
        check_shapes=[(16, 16, 129), (31, 16, 128), (129, 16, 128), (1024, 16, 128)],
        ref=torch_baseline,
        flops=lambda n, h, d: 4 * (n**2) * h * d,
        nbytes=lambda n, h, d: 4 * n * h * d * 2,
        sol={"rtx pro 4500": 219.7, "a100": 312.0, "h100": 989.0, "5090": 209.5},
        num_input_sets=4,
        # bf16 in/out, fp32 accumulate: input rounding dominates, ~2^-8 rel
        tol={"rtol": 3e-2, "atol": 2e-2},
    )
