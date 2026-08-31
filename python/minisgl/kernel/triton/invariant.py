"""Fixed reduction layouts for the experimental BF16 target runtime.

Tile sizes and reduction order never depend on the number of query rows. These
kernels trade some library-kernel throughput for decode/verify batch invariance.
They do not promise reproducibility across GPU architectures or compiler versions.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _linear(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + k
        x = tl.load(X + m[:, None] * K + kk[None, :], (m[:, None] < M) & (kk[None, :] < K), other=0)
        w = tl.load(W + n[None, :] * K + kk[:, None], (n[None, :] < N) & (kk[:, None] < K), other=0)
        partial = tl.dot(x, w)
        # Round each partial sum with FP32 add rather than carrying the entire
        # K reduction through MMA. Explicit PTX prevents the compiler folding
        # this add back into the dot accumulator (measured on SM89, K=9216).
        acc = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[acc, partial],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


def invariant_linear(x, weight, bias=None, *, fp32_output=False):
    if (
        not x.is_cuda
        or not weight.is_cuda
        or x.dtype != torch.bfloat16
        or weight.dtype != x.dtype
        or weight.ndim != 2
        or x.shape[-1] != weight.shape[1]
    ):
        raise ValueError("Invariant linear requires compatible BF16 CUDA inputs/weights")
    shape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1]).contiguous()
    weight = weight.contiguous()
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=torch.float32 if fp32_output else x.dtype)
    if m:
        _linear[(triton.cdiv(m, 16), triton.cdiv(n, 64))](
            x, weight, out, m, n, k, 16, 64, 64, num_warps=4, num_stages=3
        )
    if bias is not None:
        out = out + bias
    return out.view(*shape, n)


@triton.jit
def _attention(
    Q,
    K,
    V,
    PAGE,
    SLOTS,
    POS,
    O,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    PS: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
):
    token, h = tl.program_id(0), tl.program_id(1)
    kh = h // (HQ // HK)
    slot = tl.load(SLOTS + token)
    end = tl.load(POS + token) + 1
    ds, ns = tl.arange(0, BD), tl.arange(0, BK)
    q = tl.load(Q + token * Q0 + h * Q1 + ds, ds < D, other=0).to(tl.float32)
    q *= D**-0.5
    acc = tl.zeros((BD,), tl.float32)
    den, mx = 0.0, -float("inf")
    # A query's causal end, not the block's KV end, controls its reduction.
    for start in range(tl.cdiv(end, BK)):
        nn = start * BK + ns
        ix = tl.load(PAGE + slot * PS + nn, nn < end, other=0)
        mask = (nn[:, None] < end) & (ds[None, :] < D)
        keys = tl.load(K + (ix[:, None] * HK + kh) * D + ds[None, :], mask, other=0).to(tl.float32)
        scores = tl.sum(keys * q[None, :], axis=1)
        scores = tl.where(nn < end, scores, -float("inf"))
        new_mx = tl.maximum(mx, tl.max(scores, 0))
        alpha = tl.exp(mx - new_mx)
        prob = tl.exp(scores - new_mx)
        vals = tl.load(V + (ix[:, None] * HK + kh) * D + ds[None, :], mask, other=0).to(tl.float32)
        acc = acc * alpha + tl.sum(prob[:, None] * vals, axis=0)
        den = den * alpha + tl.sum(prob, 0)
        mx = new_mx
    tl.store(O + (token * HQ + h) * D + ds, acc / den, ds < D)


def invariant_attention(q, k_cache, v_cache, page_table, slots, positions):
    if (
        q.ndim != 3
        or q.dtype != torch.bfloat16
        or not q.is_cuda
        or not k_cache.is_contiguous()
        or not v_cache.is_contiguous()
        or k_cache.shape != v_cache.shape
        or k_cache.dtype != q.dtype
        or v_cache.dtype != q.dtype
        or q.stride(-1) != 1
    ):
        raise ValueError("Invariant attention requires BF16 CUDA queries and contiguous KV")
    count, hq, dim = q.shape
    hk = k_cache.shape[-2]
    if (
        hq % hk
        or k_cache.shape[-1] != dim
        or slots.numel() != count
        or positions.numel() != count
        or page_table.stride(-1) != 1
    ):
        raise ValueError("Incompatible invariant attention metadata")
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    if count:
        _attention[(count, hq)](
            q,
            k_cache,
            v_cache,
            page_table,
            slots,
            positions,
            out,
            hq,
            hk,
            dim,
            page_table.stride(0),
            q.stride(0),
            q.stride(1),
            32,
            triton.next_power_of_2(dim),
            num_warps=4,
        )
    return out
