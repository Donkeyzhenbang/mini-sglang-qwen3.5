from types import SimpleNamespace

import pytest
import torch
from minisgl.kernel.triton.draft_ops import (
    prepare_rms_input,
    prepare_rotary_input,
    prepare_silu_input,
)
from minisgl.kernel.triton.state_copy import LayerStateCopier, LayerStateSnapshot


def test_elementwise_pack_preserves_logical_values_of_strided_views():
    # These views have the same logical shape as contiguous inputs, but flattened
    # kernel loads without packing read different elements of their base storage.
    base = torch.arange(96, dtype=torch.float32).reshape(4, 24)
    x = base[:, 1::2]
    weight = torch.arange(24, dtype=torch.float32)[::2]
    packed, packed_weight = prepare_rms_input(x, weight, 1e-6)
    assert packed.is_contiguous() and packed_weight.is_contiguous()
    torch.testing.assert_close(packed, x, rtol=0, atol=0)
    torch.testing.assert_close(packed_weight, weight, rtol=0, atol=0)
    gate_up = prepare_silu_input(x)
    assert gate_up.is_contiguous()
    torch.testing.assert_close(gate_up, x, rtol=0, atol=0)
    # Existing contiguous hot path incurs no new tensor allocation.
    assert prepare_silu_input(packed) is packed
    assert prepare_rms_input(packed, packed_weight, 1e-6)[0] is packed


@pytest.mark.parametrize(
    "x", [torch.ones(2, 5), torch.ones(2, 0), torch.ones(2, 6, dtype=torch.long)]
)
def test_silu_rejects_unrepresentable_kernel_layout(x):
    with pytest.raises(ValueError):
        prepare_silu_input(x)


@pytest.mark.parametrize(
    "weight,eps",
    [
        (torch.ones(3), 1e-6),
        (torch.ones(4), float("nan")),
        (torch.ones(4, dtype=torch.float64), 1e-6),
    ],
)
def test_rms_rejects_wrong_weight_or_epsilon(weight, eps):
    with pytest.raises(ValueError):
        prepare_rms_input(torch.ones(2, 4), weight, eps)


def test_rotary_pack_broadcasts_positions_and_preserves_strided_table():
    x = torch.arange(96, dtype=torch.float32).reshape(2, 3, 2, 8).transpose(1, 2)
    cache = torch.arange(160, dtype=torch.float32).reshape(10, 16)[:, ::2]
    positions = torch.tensor([0, 2, 4])
    packed, pos, table = prepare_rotary_input(x, positions, cache)
    assert all(t.is_contiguous() for t in (packed, pos, table))
    torch.testing.assert_close(packed, x, rtol=0, atol=0)
    torch.testing.assert_close(table, cache, rtol=0, atol=0)
    assert pos.tolist() == [[0, 2, 4], [0, 2, 4]]
    with pytest.raises(ValueError, match="positions"):
        prepare_rotary_input(x, positions[:2], cache)
    with pytest.raises(ValueError, match="cache"):
        prepare_rotary_input(x, positions, cache[:, :4])


@pytest.mark.parametrize("slots", [[-1], [4], [1, 1], [True], [1.5], []])
def test_checkpoint_rejects_bad_slots_before_allocating_or_launching(slots):
    copier = object.__new__(LayerStateCopier)
    copier.capacity = 4
    # No device/pointer fields: bad input must fail before accessing the GPU.
    with pytest.raises(ValueError):
        copier.checkpoint(slots)


@pytest.mark.parametrize("slots,rows", [([1, 1], [0, 1]), ([4], [0]), ([1], [-1]), ([1], [2])])
def test_restore_rejects_invalid_destination_or_snapshot_row(slots, rows):
    copier = object.__new__(LayerStateCopier)
    copier.capacity = 4
    snapshot = LayerStateSnapshot(copier, torch.empty(1, 2, 3), torch.empty(1, 2, 4))
    with pytest.raises(ValueError):
        copier.restore(snapshot, slots, rows)


def test_state_capacity_mismatch_is_rejected_before_gpu_pointer_table():
    rt = SimpleNamespace(conv_cache=torch.empty(3, 2, 3), ssm_cache=torch.empty(2, 4))
    with pytest.raises(ValueError, match="capacities"):
        LayerStateCopier({0: rt})
