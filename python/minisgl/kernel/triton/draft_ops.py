"""Fused DFlash elementwise operations preserving intermediate BF16 rounding."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _rms(X, W, Y, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, other=0).to(tl.float32)
    w = tl.load(W + col, col < N, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / N + EPS)
    rounded = (x * inv).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + row * N + col, rounded * w, col < N)


def prepare_rms_input(x, weight, eps):
    """Check/pack the kernel's tensor contract without touching CUDA state."""
    if x.ndim < 1 or x.shape[-1] < 1:
        raise ValueError("RMSNorm requires a nonempty feature dimension")
    if weight.ndim != 1 or weight.numel() != x.shape[-1]:
        raise ValueError("RMSNorm weight must match the feature dimension")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("RMSNorm requires floating-point input")
    if weight.dtype != x.dtype or weight.device != x.device:
        raise ValueError("RMSNorm input and weight must share dtype and device")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("RMSNorm epsilon must be finite and positive")
    return x.contiguous(), weight.contiguous()


def rms_norm(x, weight, eps):
    x, weight = prepare_rms_input(x, weight, eps)
    if not x.is_cuda:
        raise ValueError("Fused RMSNorm requires CUDA tensors")
    out = torch.empty_like(x)
    if not x.numel():
        return out
    n = x.shape[-1]
    _rms[(x.numel() // n,)](
        x, weight, out, N=n, EPS=eps, BLOCK=triton.next_power_of_2(n), enable_fp_fusion=False
    )
    return out


@triton.jit
def _silu_mul(X, Y, N: tl.constexpr, TOTAL: tl.constexpr, TILE: tl.constexpr):
    idx = tl.program_id(0) * TILE + tl.arange(0, TILE)
    src = idx // N * (2 * N) + idx % N
    gate = tl.load(X + src, idx < TOTAL, other=0).to(tl.float32)
    up = tl.load(X + src + N, idx < TOTAL, other=0).to(tl.float32)
    activated = (gate * tl.sigmoid(gate)).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + idx, activated * up, idx < TOTAL)


def prepare_silu_input(x):
    if x.ndim < 1 or x.shape[-1] < 2 or x.shape[-1] % 2:
        raise ValueError("SiLU multiply requires a positive even gate/up width")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("SiLU multiply requires floating-point input")
    # The Triton kernel uses flattened pointer arithmetic, not tensor strides.
    return x.contiguous()


def silu_mul(x):
    x = prepare_silu_input(x)
    if not x.is_cuda:
        raise ValueError("Fused SiLU multiply requires a CUDA tensor")
    n = x.shape[-1] // 2
    out = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
    if not out.numel():
        return out
    _silu_mul[(triton.cdiv(out.numel(), 256),)](
        x, out, N=n, TOTAL=out.numel(), TILE=256, enable_fp_fusion=False
    )
    return out


@triton.jit
def _rope(
    X,
    POS,
    CACHE,
    Y,
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    TOTAL: tl.constexpr,
    TILE: tl.constexpr,
    CAPACITY: tl.constexpr,
):
    idx = tl.program_id(0) * TILE + tl.arange(0, TILE)
    dim = idx % D
    row = idx // D
    token, batch = row % T, row // (H * T)
    position = tl.load(POS + batch * T + token, idx < TOTAL, other=0)
    x = tl.load(X + idx, idx < TOTAL, other=0).to(tl.float32)
    half = D // 2
    rotated_dim = (dim + half) % D
    other = tl.load(X + row * D + rotated_dim, idx < TOTAL, other=0).to(tl.float32)
    rotated = tl.where(dim < half, -other, other)
    cos = tl.load(
        CACHE + position * D + dim % half,
        (idx < TOTAL) & (position >= 0) & (position < CAPACITY),
        other=1,
    ).to(tl.float32)
    sin = tl.load(
        CACHE + position * D + dim % half + half,
        (idx < TOTAL) & (position >= 0) & (position < CAPACITY),
        other=0,
    ).to(tl.float32)
    first = (x * cos).to(Y.dtype.element_ty).to(tl.float32)
    second = (rotated * sin).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + idx, first + second, idx < TOTAL)


def prepare_rotary_input(x, positions, cache):
    if x.ndim != 4 or x.shape[-1] < 2 or x.shape[-1] % 2:
        raise ValueError("Cached rotary requires [batch, heads, tokens, even head_dim]")
    batch, heads, count, dim = x.shape
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Cached rotary requires floating-point input")
    if cache.ndim != 2 or cache.shape[1] != dim or cache.shape[0] < 1:
        raise ValueError("Rotary cache must have shape [capacity, head_dim]")
    if cache.device != x.device or cache.dtype != x.dtype:
        raise ValueError("Rotary cache must share input dtype and device")
    if positions.device != x.device or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("Rotary positions must be integer tensors on the input device")
    if positions.ndim == 1:
        if positions.shape[0] != count:
            raise ValueError("Rotary positions must match token count")
        positions = positions.expand(batch, -1)
    if positions.shape != (batch, count):
        raise ValueError("Rotary positions must match [batch, tokens]")
    return x.contiguous(), positions.contiguous(), cache.contiguous()


def cached_rotary(x, positions, cache):
    # Out-of-table positions are identity-rotated padding, including negative
    # sentinels. Callers must mask padding out of attention; this is not RoPE
    # extrapolation for real tokens. No GPU -> CPU bounds check during capture.
    x, positions, cache = prepare_rotary_input(x, positions, cache)
    if not x.is_cuda:
        raise ValueError("Fused cached rotary requires CUDA tensors")
    out = torch.empty_like(x)
    if not x.numel():
        return out
    batch, heads, count, dim = x.shape
    _rope[(triton.cdiv(x.numel(), 256),)](
        x,
        positions,
        cache,
        out,
        H=heads,
        T=count,
        D=dim,
        TOTAL=x.numel(),
        TILE=256,
        CAPACITY=cache.shape[0],
        enable_fp_fusion=False,
    )
    return out
