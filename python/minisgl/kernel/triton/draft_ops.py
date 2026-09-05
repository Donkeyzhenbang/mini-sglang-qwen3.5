"""Fused DFlash elementwise operations preserving intermediate BF16 rounding."""

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


def rms_norm(x, weight, eps):
    x = x.contiguous()
    out = torch.empty_like(x)
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


def silu_mul(x):
    n = x.shape[-1] // 2
    out = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
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
        CACHE + position * D + dim % half, (idx < TOTAL) & (position < CAPACITY), other=1
    ).to(tl.float32)
    sin = tl.load(
        CACHE + position * D + dim % half + half, (idx < TOTAL) & (position < CAPACITY), other=0
    ).to(tl.float32)
    first = (x * cos).to(Y.dtype.element_ty).to(tl.float32)
    second = (rotated * sin).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + idx, first + second, idx < TOTAL)


def cached_rotary(x, positions, cache):
    x = x.contiguous()
    out = torch.empty_like(x)
    batch, heads, count, dim = x.shape
    if positions.ndim == 1:
        positions = positions.expand(batch, -1)
    positions = positions.contiguous()
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
