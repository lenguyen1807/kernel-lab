import torch
import torch.nn.functional as F


def attention_naive(q, k, v, causal_mask: bool = False):
    """
    q: [1, H, N, d]
    k: [1, H, N, d]
    v: [1, H, N, d]
    """
    scaled_qkt = torch.einsum("bhnd,bhmd->bhnm", q, k) / (k.shape[-1] ** 0.5)
    if causal_mask:
        scaled_qkt = causal_mask(scaled_qkt)
    score = F.softmax(scaled_qkt, dim=-1)
    return torch.einsum("bhmn,bhnd->bhmd", score, v)


def causal_mask(qkt):
    all_ones = torch.ones(qkt.shape[-2], qkt.shape[-1], device=qkt.device, dtype=torch.bool)
    mask = torch.triu(all_ones, diagonal=1)
    ignore = torch.tensor(float("-inf"), dtype=torch.float32, device=qkt.device)
    qkt.masked_fill_(mask, ignore)
    return qkt
