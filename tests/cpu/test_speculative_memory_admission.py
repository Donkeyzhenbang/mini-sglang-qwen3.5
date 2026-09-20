from types import SimpleNamespace

import pytest
import torch
from minisgl.runtime.adaptive import JointMemoryBudget
from minisgl.speculative.target import MiniSGLTarget


@pytest.mark.parametrize(
    "allocated,reserved,free,capacity",
    [
        (1 << 30, 2 << 30, 0, 4 << 30),
        (1 << 30, 1 << 30, 2 << 30, 4 << 30),
        (1 << 30, 2 << 30, 2 << 30, (1 << 30) + 1),
        (1 << 30, 1 << 30, 0, 4 << 30),
        (1 << 30, (1 << 30) + (257 << 20), 1 << 30, 4 << 30),
    ],
)
def test_allocator_admission_matches_driver_oracle(
    monkeypatch, allocated, reserved, free, capacity
):
    target = MiniSGLTarget.__new__(MiniSGLTarget)
    target.device, target.slot, target.executor = "cuda", 0, object()
    target.budget_bytes, target.safety_bytes = capacity, 256 << 20
    target.cache = None
    target.embedding = torch.empty(1000, 16, dtype=torch.bfloat16)
    target.gdn = SimpleNamespace(
        _runtime={
            0: SimpleNamespace(
                conv_cache=torch.empty(4, 16),
                ssm_cache=torch.empty(4, 128),
            )
        }
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda device: {
            "allocated_bytes.all.current": allocated,
            "reserved_bytes.all.current": reserved,
        },
    )
    queries = []

    def query(device):
        queries.append(device)
        return free, 24 << 30

    monkeypatch.setattr(torch.cuda, "mem_get_info", query)
    blocks, context, batch = [1, 2, 4, 8, 16], 4096, 4
    per_token = 8 * 1000 + 32 * 16 * 2 + 384 * context
    checkpoint = (16 + 128) * 4 * batch * 2
    budget = JointMemoryBudget(capacity, 0, target.safety_bytes)

    def oracle(available):
        return budget.feasible_blocks(
            blocks,
            live_bytes=max(0, capacity - available),
            bytes_per_block_token=per_token * batch,
            checkpoint_bytes=checkpoint,
        )

    expected = oracle(min(free + reserved - allocated, capacity - allocated))
    pooled = oracle(min(reserved - allocated, capacity - allocated))
    assert target.feasible_blocks(context, blocks, batch) == expected
    assert len(queries) == (0 if pooled == blocks else 1)
