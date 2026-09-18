"""Gather/scatter homogeneous per-layer states without relocating live graph buffers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


def validate_state_indices(indices, capacity, *, unique, label):
    if any(not isinstance(i, int) or isinstance(i, bool) for i in indices):
        raise ValueError(f"{label} must contain host integer indices")
    if any(i < 0 or i >= capacity for i in indices):
        raise ValueError(f"{label} index out of range")
    if unique and len(set(indices)) != len(indices):
        raise ValueError(f"{label} must be unique to avoid concurrent state writes")


@triton.jit
def _copy_slots(
    PTRS,
    SLOTS,
    ROWS,
    DATA,
    N: tl.constexpr,
    BATCH: tl.constexpr,
    RESTORE: tl.constexpr,
    TILE: tl.constexpr,
):
    layer, request = tl.program_id(1), tl.program_id(2)
    offset = tl.program_id(0) * TILE + tl.arange(0, TILE)
    slot, row = tl.load(SLOTS + request), tl.load(ROWS + request)
    source = tl.load(PTRS + layer).to(tl.pointer_type(DATA.dtype.element_ty))
    live = source + slot * N + offset
    saved = DATA + (layer * BATCH + row) * N + offset
    if RESTORE:
        value = tl.load(saved, mask=offset < N, other=0)
        tl.store(live, value, mask=offset < N)
    else:
        value = tl.load(live, mask=offset < N, other=0)
        tl.store(saved, value, mask=offset < N)


@dataclass
class LayerStateSnapshot:
    copier: "LayerStateCopier"
    conv: torch.Tensor
    ssm: torch.Tensor


class LayerStateCopier:
    """Two launches for all layers; snapshots own their data until rollback ends.

    Pointer tables hold strong references to the original state allocations.
    Existing CUDA decode graphs keep exactly the same state addresses.
    Host-side slot/row validation prevents out-of-bounds or racing kernel writes.
    """

    def __init__(self, runtime):
        self.layers = tuple(runtime)
        self.states = tuple((rt.conv_cache, rt.ssm_cache) for rt in runtime.values())
        if not self.states:
            raise ValueError("Cannot pack an empty state runtime")
        if any(t.ndim < 2 or t.shape[0] < 1 for pair in self.states for t in pair):
            raise ValueError("State packing requires nonempty slot-major tensors")
        self.capacity = self.states[0][0].shape[0]
        if any(t.shape[0] != self.capacity for pair in self.states for t in pair):
            raise ValueError("Conv and SSM state slot capacities must match")
        for kind in range(2):
            first = self.states[0][kind]
            if any(
                not t.is_cuda
                or not t.is_contiguous()
                or t.shape != first.shape
                or t.dtype != first.dtype
                or t.device != first.device
                for t in (pair[kind] for pair in self.states)
            ):
                raise ValueError("State packing requires homogeneous contiguous CUDA layers")
        self.device = self.states[0][0].device
        if self.states[0][1].device != self.device:
            raise ValueError("Conv and SSM states must share a CUDA device")
        self.pointers = tuple(
            torch.tensor(
                [pair[kind].data_ptr() for pair in self.states],
                device=self.device,
                dtype=torch.int64,
            )
            for kind in range(2)
        )

    def matches(self, runtime):
        return self.layers == tuple(runtime) and all(
            conv is rt.conv_cache and ssm is rt.ssm_cache
            for (conv, ssm), rt in zip(self.states, runtime.values())
        )

    def checkpoint(self, slots):
        validate_state_indices(slots, self.capacity, unique=True, label="State slots")
        if not slots:
            raise ValueError("Cannot checkpoint an empty slot set")
        batch = len(slots)
        metadata = torch.tensor([slots, list(range(batch))], dtype=torch.int64, device=self.device)
        saved = tuple(
            torch.empty(
                (len(self.states), batch, pair.numel() // pair.shape[0]),
                dtype=pair.dtype,
                device=self.device,
            )
            for pair in self.states[0]
        )
        for pointers, data in zip(self.pointers, saved):
            _copy_slots[(triton.cdiv(data.shape[2], 4096), len(self.states), batch)](
                pointers,
                metadata[0],
                metadata[1],
                data,
                N=data.shape[2],
                BATCH=batch,
                RESTORE=False,
                TILE=4096,
            )
        return LayerStateSnapshot(self, *saved)

    def restore(self, snapshot, slots, rows):
        if snapshot.copier is not self or len(slots) != len(rows):
            raise ValueError("State snapshot does not match the copier")
        validate_state_indices(slots, self.capacity, unique=True, label="State slots")
        validate_state_indices(rows, snapshot.conv.shape[1], unique=False, label="Snapshot rows")
        if not slots:
            return
        metadata = torch.tensor([slots, rows], dtype=torch.int64, device=self.device)
        for pointers, data in zip(self.pointers, (snapshot.conv, snapshot.ssm)):
            _copy_slots[(triton.cdiv(data.shape[2], 4096), len(self.states), len(slots))](
                pointers,
                metadata[0],
                metadata[1],
                data,
                N=data.shape[2],
                BATCH=data.shape[1],
                RESTORE=True,
                TILE=4096,
            )
