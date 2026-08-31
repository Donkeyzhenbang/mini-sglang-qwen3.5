"""Experimental packed recurrent GDN extend: one launch per layer.

Unlike the Python token loop, recurrence stays on device. This is not the
chunk-parallel WY algorithm and must be benchmarked before changing defaults.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _extend_kernel(
    QKV,
    A,
    B,
    ALOG,
    DT,
    STATE,
    SLOTS,
    CU,
    OUT,
    D: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    tile, nh = tl.program_id(0), tl.program_id(1)
    req, vh = nh // HV, nh % HV
    qh = vh // (HV // H)
    kk = tl.arange(0, BK)
    vv = tile * BV + tl.arange(0, BV)
    mask = (vv[:, None] < V) & (kk[None, :] < K)
    slot = tl.load(SLOTS + req)
    start, end = tl.load(CU + req), tl.load(CU + req + 1)
    if slot < 0:
        for token in range(start, end):
            tl.store(OUT + (token * HV + vh) * V + vv, 0.0, mask=vv < V)
        return
    address = STATE + ((slot * HV + vh) * V + vv[:, None]) * K + kk[None, :]
    state = tl.load(address, mask=mask, other=0).to(tl.float32)
    alog, dt = tl.load(ALOG + vh).to(tl.float32), tl.load(DT + vh).to(tl.float32)
    for token in range(start, end):
        base = QKV + token * D
        q = tl.load(base + qh * K + kk, mask=kk < K, other=0).to(tl.float32)
        k = tl.load(base + H * K + qh * K + kk, mask=kk < K, other=0).to(tl.float32)
        v = tl.load(base + 2 * H * K + vh * V + vv, mask=vv < V, other=0).to(tl.float32)
        # Preserve the same operation order as the one-token decode kernel.
        q = q / tl.sqrt(tl.sum(q * q) + 1e-6)
        k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
        q = q * (K**-0.5)
        a = tl.load(A + token * HV + vh).to(tl.float32) + dt
        beta = tl.sigmoid(tl.load(B + token * HV + vh).to(tl.float32))
        beta = beta.to(B.dtype.element_ty).to(tl.float32)
        softplus = tl.where(a <= 20, tl.log(1 + tl.exp(a)), a)
        state *= tl.exp(-tl.exp(alog) * softplus)
        delta = (v - tl.sum(state * k[None, :], axis=1)) * beta
        state += delta[:, None] * k[None, :]
        out = tl.sum(state * q[None, :], axis=1)
        tl.store(OUT + (token * HV + vh) * V + vv, out, mask=vv < V)
    tl.store(address, state, mask=mask)


def packed_extend(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    state,
    state_indices,
    cu_seqlens,
    num_q_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
):
    tensors = (mixed_qkv, a, b, A_log, dt_bias, state, state_indices, cu_seqlens)
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        raise ValueError("Packed extend requires contiguous CUDA tensors")
    if num_v_heads % num_q_heads or state.ndim != 4:
        raise ValueError("Invalid GDN head/state layout")
    if len(cu_seqlens) != len(state_indices) + 1:
        raise ValueError("Packed sequence metadata mismatch")
    out = torch.empty(
        (len(mixed_qkv), num_v_heads, head_v_dim), device=mixed_qkv.device, dtype=mixed_qkv.dtype
    )
    bv = min(triton.next_power_of_2(head_v_dim), 32)
    _extend_kernel[(triton.cdiv(head_v_dim, bv), len(state_indices) * num_v_heads)](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        state,
        state_indices,
        cu_seqlens,
        out,
        D=mixed_qkv.shape[1],
        H=num_q_heads,
        HV=num_v_heads,
        K=head_k_dim,
        V=head_v_dim,
        BK=triton.next_power_of_2(head_k_dim),
        BV=bv,
        num_warps=1,
    )
    return out
