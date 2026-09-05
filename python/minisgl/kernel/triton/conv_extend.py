"""Short ragged causal convolution with one launch across request slots.

Each program owns a request/channel tile, reading all old history before
writing its final history. No other program can race on that history tile.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _conv_extend(
    X,
    W,
    STATE,
    SLOTS,
    CU,
    ENDS,
    Y,
    SELECTIVE: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    TILE: tl.constexpr,
):
    channel = tl.program_id(0) * TILE + tl.arange(0, TILE)
    req = tl.program_id(1)
    valid = channel < D
    slot = tl.load(SLOTS + req)
    start = tl.load(CU + req)
    end = tl.load(ENDS + req) if SELECTIVE else tl.load(CU + req + 1)
    hist = tl.arange(0, 4)
    state_mask = valid[:, None] & (hist[None, :] < K - 1)
    history = tl.load(
        STATE + (slot * D + channel[:, None]) * (K - 1) + hist[None, :], mask=state_mask, other=0
    )
    for token in range(start, end):
        acc = tl.full((TILE,), 0, tl.float32)
        for tap in tl.static_range(K):
            pos = token - start + tap - (K - 1)
            old = tl.sum(tl.where(hist[None, :] == pos + K - 1, history.to(tl.float32), 0), axis=1)
            value = tl.load(X + (start + pos) * D + channel, mask=valid & (pos >= 0), other=0).to(
                tl.float32
            )
            value = tl.where(pos >= 0, value, old)
            weight = tl.load(W + channel * K + tap, mask=valid, other=0).to(tl.float32)
            acc = acc + value * weight
        # Match torch conv1d BF16 rounding before SiLU.
        rounded = acc.to(X.dtype.element_ty).to(tl.float32)
        y = rounded * tl.sigmoid(rounded)
        tl.store(Y + token * D + channel, y, mask=valid)
    pos = end - start - (K - 1) + hist
    old_pos = pos + K - 1
    # K=4: select old history explicitly without reading history being updated.
    updated = tl.full((TILE, 4), 0, tl.float32)
    for tap in tl.static_range(K - 1):
        old = tl.sum(tl.where(hist[None, :] == tap, history.to(tl.float32), 0), axis=1)
        updated += tl.where(old_pos[None, :] == tap, old[:, None], 0)
    recent = tl.load(
        X + (start + pos[None, :]) * D + channel[:, None],
        mask=state_mask & (pos[None, :] >= 0),
        other=0,
    )
    updated = tl.where(pos[None, :] >= 0, recent, updated)
    tl.store(
        STATE + (slot * D + channel[:, None]) * (K - 1) + hist[None, :], updated, mask=state_mask
    )


def packed_conv_extend(x, weight, state, slots, cu, *, end_offsets=None):
    tensors = (x, weight, state, slots, cu)
    if end_offsets is not None:
        tensors += (end_offsets,)
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        raise ValueError("Packed convolution requires contiguous CUDA tensors")
    if weight.shape[-1] != 4 or state.shape[-1] != 3:
        raise ValueError("Packed convolution currently supports kernel width 4")
    if len(cu) != len(slots) + int(end_offsets is None) or (
        end_offsets is not None and len(end_offsets) != len(slots)
    ):
        raise ValueError("Packed sequence metadata mismatch")
    metadata = (slots, cu) + (() if end_offsets is None else (end_offsets,))
    if any(t.ndim != 1 or t.dtype not in (torch.int32, torch.int64) for t in metadata):
        raise ValueError("Packed metadata must be one-dimensional integer tensors")
    out = torch.empty_like(x)
    _conv_extend[(triton.cdiv(x.shape[1], 128), len(slots))](
        x,
        weight,
        state,
        slots,
        cu,
        end_offsets if end_offsets is not None else cu,
        out,
        SELECTIVE=end_offsets is not None,
        D=x.shape[1],
        K=4,
        TILE=128,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
