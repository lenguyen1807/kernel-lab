import torch
import triton
import triton.language as tl
import math

@triton.jit
def flash_attention_1_kernel(
    Q, K, V, O,
    N, d, # N is the number of tokens, d is the dimension of the embedding
    stride_qn, stride_qd,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    stride_on, stride_od,
    l_ptr, m_ptr, # l is accumulation of total sum, m is current maximum value
    B_c: tl.constexpr,
    B_r: tl.constexpr,
    T_r: tl.constexpr,
    T_c: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offset_d = tl.arange(0, d)
    offset_qo = pid_n * B_r + tl.arange(0, B_r)
    offset_kv = tl.arange(0, B_c)

    q_tile = Q + (offset_qo[:, None] * stride_qn + offset_d[None, :] * stride_qd)
    k_tile = K + (offset_kv[:, None] * stride_kn + offset_d[None, :] * stride_kd)
    v_tile = V + (offset_kv[:, None] * stride_vn + offset_d[None, :] * stride_vd)
    o_tile = O + (offset_qo[:, None] * stride_on + offset_d[None, :] * stride_od)

    s_tile = tl.zeros((B_r, d), dtype=tl.float32)

    for i in range(0, T_c):
        # load k_i and v_i
        k_i = tl.load(k_tile, mask=offset_kv[None, :] < N - i * B_c, other=0.0)
        v_i = tl.load(v_tile, mask=offset_kv[None, :] < N - i * B_c, other=0.0)

        for j in range(0, T_r):
            q_j = tl.load(q_tile, mask=offset_qo[:, None] < N - j * B_r, other=0.0)
            o_j = tl.load(o_tile, mask=offset_qo[:, None] < N - j * B_r, other=0.0)
            l_j = tl.load(l_ptr + offset_qo, mask=offset_qo < N, other=0.0)
            m_j = tl.load(m_ptr + offset_qo, mask=offset_qo < N, other=float("-inf"))

            s_ij = tl.dot(q_j, k_i.T)
            m_ij = tl.max(m_j, s_ij, axis=1)
            p_ij = tl.exp(s_ij - m_ij)
            l_ij = tl.sum(p_ij, axis=1)
            
            m_new = tl.maximum(m_j, m_ij)
            l_new = tl.exp(m_j - m_new) * l_j + tl.exp(m_ij - m_new) * l_ij

            tl.store(l_ptr + offset_qo, l_new, mask=offset_qo < N)
            tl.store(m_ptr + offset_qo, m_new, mask=offset_qo < N)


def flash_attention_1(Q, K, V, causal_mask: bool = False):
    """
    q: [1, H, N, d]
    k: [1, H, N, d]
    v: [1, H, N, d]
    """
    assert Q.is_cuda
    assert K.device == Q.device
    assert V.device == Q.device

    # first, we remove batch dimension
    Q = Q.squeeze(0)
    K = K.squeeze(0)
    V = V.squeeze(0)
    H, N, d = Q.shape

    O = torch.zeros_like(Q)
    l = torch.zeros(N, device=Q.device, dtype=Q.dtype)
    # we need m in float32 to handle -inf
    m = torch.full(N, float("-inf"), device=Q.device, dtype=torch.float32)

    grid = lambda meta: (
        H, # head
        triton.cdiv(N, meta["BLOCK_SIZE_r"])
    )

    # we assume the shared memory capability is 128KiB
    # each element is 2 bytes (bfloat16)
    # so we can use (128 * 1024) / 2 = 65536

    M = 65536
    B_r = math.ceil(M / (4 * d))
    B_c = math.min(d, B_r)
    T_r = math.ceil(N / B_r)
    T_c = math.ceil(N / B_c)

    flash_attention_1_kernel[grid](
        Q, K, V, O,
        N, d,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        O.stride(0), O.stride(1),
        l, m,
        B_c=B_c, B_r=B_r, T_r=T_r, T_c=T_c
    )
    
    return O