from types import SimpleNamespace

import pytest
import torch
from minisgl.speculative.graph_contract import (
    cache_import_required,
    can_capture_graph,
    validate_graph_requests,
)


def test_same_storage_different_slot_is_not_a_valid_cache_hit():
    pool = torch.arange(3 * 2 * 8 * 4).reshape(3, 2, 8, 4)
    destination = pool[1:2, :, :5]
    stale = pool[0:1, :, :5]
    assert stale.untyped_storage().data_ptr() == destination.untyped_storage().data_ptr()
    assert not torch.equal(stale, destination)
    with pytest.raises(ValueError, match="different graph slot"):
        cache_import_required(stale, destination)
    assert not cache_import_required(pool[1:2, :, :5], destination)
    assert cache_import_required(stale.clone(), destination)


def test_cache_contract_rejects_stale_lengths_and_allows_empty_reset():
    pool = torch.zeros(2, 2, 8, 4)
    destination = pool[1:2, :, 3:7]
    assert not cache_import_required(destination, destination)
    with pytest.raises(ValueError, match="mismatch"):
        cache_import_required(pool[1:2, :, :3], destination)
    with pytest.raises(ValueError, match="mismatch"):
        cache_import_required(destination.double(), destination)
    with pytest.raises(ValueError, match="Missing"):
        cache_import_required(None, destination)
    assert not cache_import_required(None, pool[1:2, :, :0])


@pytest.mark.parametrize("slots", [[], [0], [1, 1], [-1, 1], [0, 4], [True, 2], [0.5, 2]])
def test_request_slots_are_checked_before_graph_work(slots):
    with pytest.raises(ValueError):
        validate_graph_requests([(object(),), (object(),)], slots, 4)


def test_graph_allows_request_reordering_but_rejects_duplicate_context():
    a, b = object(), object()
    validate_graph_requests([(b,), (a,)], [3, 0], 4)
    with pytest.raises(ValueError, match="distinct"):
        validate_graph_requests([(a,), (a,)], [0, 1], 4)


def test_graph_admission_preserves_existing_shapes_at_memory_pressure():
    graphs = {(1, 4): object()}
    assert can_capture_graph(graphs, (1, 4), free_bytes=0, limit=1)
    assert not can_capture_graph(graphs, (2, 4), free_bytes=8 * 2**30, limit=1)
    assert not can_capture_graph(graphs, (2, 4), free_bytes=2**30)
    assert can_capture_graph(graphs, (2, 4), free_bytes=2 * 2**30)


def test_journal_rejects_duplicate_state_writes_before_replay():
    from minisgl.attention.gdn import GDNAttnBackend

    backend = GDNAttnBackend()
    backend._runtime = {0: SimpleNamespace(ssm_cache=torch.empty(4, 2))}
    backend._verify_journal = {0: object()}
    with pytest.raises(ValueError, match="unique"):
        backend.commit_verify_journal([(SimpleNamespace(slot=1), 1), (SimpleNamespace(slot=1), 2)])


@pytest.mark.parametrize("kind", ["dflash", "mtp"])
def test_pool_validates_whole_import_batch_before_mutation(kind, monkeypatch):
    from minisgl.speculative.draft_graph import DFlashGraphPool
    from minisgl.speculative.mtp_graph import MTPGraphPool

    cls = DFlashGraphPool if kind == "dflash" else MTPGraphPool
    pool = object.__new__(cls)
    pool.embedding = torch.zeros(32, 8)
    pool.max_context, pool.context_width = 64, 0
    pool.graphs, pool.fallbacks = {}, 0
    fc = SimpleNamespace(weight=torch.empty(8, 8), in_features=8)
    pool.model = SimpleNamespace(fc=fc, block_size=4)
    shape = (1, 2, 2, 64, 4) if kind == "dflash" else (2, 2, 64, 4)
    pool.keys = torch.zeros(shape)
    pool.values = torch.zeros(shape)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (8 * 2**30, 16 * 2**30))
    items = []
    for slot in range(2):
        # First request would import ones; second has a stale cache length.
        cache = torch.ones(1, 2, 2 if slot == 0 else 1, 4)
        draft = SimpleNamespace(fc=fc, context_length=2, max_steps=3)
        if kind == "dflash":
            attn = SimpleNamespace(cached_k=cache, cached_v=cache.clone(), window=None)
            draft.layers = [SimpleNamespace(self_attn=attn)]
            items.append((draft, torch.zeros(1, 1, 8), 1, 4, 3))
        else:
            draft.cached_k, draft.cached_v = cache, cache.clone()
            items.append((draft, torch.zeros(1, 8), [1], 4, 3))
    with pytest.raises(ValueError, match="mismatch"):
        pool.propose(items, [0, 1])
    assert torch.count_nonzero(pool.keys) == torch.count_nonzero(pool.values) == 0
    assert not pool.graphs


@pytest.mark.parametrize("kind", ["dflash", "mtp"])
def test_pool_rejects_same_allocation_wrong_slot_before_replay(kind):
    from minisgl.speculative.draft_graph import DFlashGraphPool
    from minisgl.speculative.mtp_graph import MTPGraphPool

    cls = DFlashGraphPool if kind == "dflash" else MTPGraphPool
    pool = object.__new__(cls)
    pool.embedding = torch.zeros(32, 8)
    pool.max_context, pool.context_width = 64, 0
    fc = SimpleNamespace(weight=torch.empty(8, 8), in_features=8)
    pool.model = SimpleNamespace(fc=fc, block_size=4)
    base = torch.zeros(2, 2, 64, 4)
    base[0].fill_(7)
    pool.keys = base.unsqueeze(0) if kind == "dflash" else base
    pool.values = pool.keys.clone()
    draft = SimpleNamespace(fc=fc, context_length=2, max_steps=3)

    def unexpected_replay(*args):
        raise AssertionError("Wrong-slot cache reached GPU graph replay")

    graph = SimpleNamespace(replay=unexpected_replay)
    if kind == "dflash":
        attn = SimpleNamespace(
            cached_k=pool.keys[0, 0:1, :, :2], cached_v=pool.values[0, 0:1, :, :2], window=None
        )
        draft.layers = [SimpleNamespace(self_attn=attn)]
        row = (draft, torch.zeros(1, 1, 8), 1, 4, 3)
        pool.graphs = {(1, 4, 4, 64): graph}
    else:
        draft.cached_k, draft.cached_v = pool.keys[0:1, :, :2], pool.values[0:1, :, :2]
        row = (draft, torch.zeros(1, 8), [1], 4, 3)
        pool.graphs = {(1, 4, 1, 64): graph}
    with pytest.raises(ValueError, match="different graph slot"):
        pool.propose([row], [1])
