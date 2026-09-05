import pytest
import torch
from minisgl.runtime.adaptive import AdaptiveBlockController, JointMemoryBudget
from minisgl.runtime.hybrid_cache import HybridPrefixCache


def test_bundle_atomic_copy_and_longest_prefix():
    c = HybridPrefixCache(0, 128)
    payload = {"kv": torch.arange(4), "ssm": torch.ones(4)}
    assert c.put([1, 2], payload, 100)
    payload["ssm"].zero_()
    hit = c.lookup([1, 2, 3])
    assert torch.all(hit.tensors["ssm"] == 1)
    assert set(hit.tensors) == {"kv", "ssm"}
    assert c.lookup([1, 3]) is None
    assert c.used("cpu") <= 128


def test_cost_eviction_preserves_expensive_prefix():
    c = HybridPrefixCache(0, 64, policy="cost")
    for token, cost in [(1, 1000), (2, 1), (3, 5)]:
        c.put([token], {"state": torch.zeros(8)}, cost)
    assert (1,) in c.entries and (2,) not in c.entries
    assert c.used("cpu") == 64
    assert not c.put([4], {"state": torch.zeros(100)}, 1)


def test_recompute_when_transfer_cost_exceeds_compute():
    c = HybridPrefixCache(0, 128, host_bandwidth_bytes_per_ms=1)
    c.put([1], {"state": torch.zeros(16)}, 1)
    assert c.lookup([1, 2]) is None
    assert c.stats["recompute_choices"] == 1


def test_controller_uses_progress_over_total_cost_not_acceptance_alone():
    c = AdaptiveBlockController(exploration_interval=100)
    for b, progress, cost in [(1, 1, 5), (2, 2, 6), (4, 4, 7), (8, 8, 50), (16, 16, 100)]:
        c.observe(
            b,
            batch_size=1,
            context_len=1024,
            progress=progress,
            draft_ms=cost / 2,
            verify_ms=cost / 2,
        )
    assert c.choose(batch_size=1, context_len=1024) == 4
    assert c.choose(batch_size=1, context_len=1024, feasible_blocks=[1, 2]) == 2
    assert c.choose(batch_size=1, context_len=1024, remaining=1) == 1


def test_adaptation_to_rejection_and_restore_cost():
    c = AdaptiveBlockController(alpha=1, exploration_interval=100)
    for b in c.blocks:
        c.observe(b, batch_size=1, context_len=1024, progress=b, draft_ms=1, verify_ms=2)
    assert c.choose(batch_size=1, context_len=1024) == 16
    c.observe(
        16, batch_size=1, context_len=1024, progress=1, draft_ms=5, verify_ms=10, restore_ms=20
    )
    assert c.choose(batch_size=1, context_len=1024) == 8


def test_memory_pressure_removes_large_blocks_and_checkpoint():
    budget = JointMemoryBudget(1000, 400, 100)
    assert budget.feasible_blocks(
        [1, 2, 4, 8, 16], live_bytes=200, bytes_per_block_token=40, checkpoint_bytes=100
    ) == [1, 2, 4]
    assert budget.feasible_blocks(
        [1, 2, 4], live_bytes=450, bytes_per_block_token=40, checkpoint_bytes=100
    ) == [1]
    assert budget.feasible_blocks([1], live_bytes=501, bytes_per_block_token=40) == []
    with pytest.raises(MemoryError):
        AdaptiveBlockController().choose(batch_size=1, context_len=1024, feasible_blocks=[])


def test_adaptive_does_not_train_steady_cost_from_graph_startup():
    from minisgl.runtime.adaptive import AdaptiveBlockController

    c = AdaptiveBlockController((1, 4), exploration_interval=100)
    c.observe(1, batch_size=4, context_len=100, progress=4, draft_ms=0, verify_ms=12)
    assert c.choose(batch_size=4, context_len=100) == 4
    c.observe(
        4, batch_size=4, context_len=100, progress=10, draft_ms=1000, verify_ms=20, startup=True
    )
    # Retry the cold candidate rather than permanently rating its capture cost.
    assert c.choose(batch_size=4, context_len=100) == 4
    c.observe(4, batch_size=4, context_len=100, progress=10, draft_ms=6, verify_ms=12)
    assert c.choose(batch_size=4, context_len=100) == 4
    assert c.startup_observations == 1
