from types import SimpleNamespace

import pytest
import torch
from minisgl import core
from minisgl.attention.gdn import GDNAttnBackend, HybridLinearBackend, _LayerRuntime
from minisgl.core import Batch, Context, Req, SamplingParams
from minisgl.kvcache.mha_pool import MHAKVCache
from minisgl.runtime.memory import HybridMemoryLayout, plan_kv_pages


@pytest.fixture
def context(monkeypatch):
    ctx = Context(256)
    ctx.page_table = torch.zeros((2, 512), dtype=torch.int32)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    return ctx


class Node:
    pass


def backend_with_state():
    b = GDNAttnBackend()
    b._runtime[0] = _LayerRuntime(torch.zeros(2, 3, 3), torch.zeros(2, 1, 2, 2))
    return b


def request(length):
    return Req(torch.arange(length), 0, 0, 10, 1, SamplingParams(), None)


def test_unaligned_prefix_is_not_published(context):
    b = backend_with_state()
    req = request(300)
    batch = Batch([req], "prefill")
    b.capture_prefix_states(batch)
    handle = SimpleNamespace(node=Node(), cached_len=256)
    b.on_prefix_cache_store(req, handle, batch.prefix_states.get(req))
    assert not b.has_prefix_cache_state(handle)


def test_snapshot_is_immutable_and_restore_is_deferred(context):
    b = backend_with_state()
    req = request(256)
    b._runtime[0].ssm_cache[0].fill_(256)
    batch = Batch([req], "prefill")
    b.capture_prefix_states(batch)
    # Simulate overlap: subsequent decode updates the live state before publication.
    b._runtime[0].ssm_cache[0].fill_(257)
    handle = SimpleNamespace(node=Node(), cached_len=256)
    b.on_prefix_cache_store(req, handle, batch.prefix_states[req])
    b.on_table_slot_allocated(1)
    b.on_prefix_cache_match(handle, 1)
    assert b._runtime[0].ssm_cache[1].sum() == 0
    b.prepare_state_slots()
    assert torch.all(b._runtime[0].ssm_cache[1] == 256)
    assert not b.has_prefix_cache_state(SimpleNamespace(node=handle.node, cached_len=255))


def test_slot_clear_does_not_race_previous_forward(context):
    b = backend_with_state()
    b._runtime[0].ssm_cache[0].fill_(9)
    b.on_table_slot_allocated(0)
    assert torch.all(b._runtime[0].ssm_cache[0] == 9)
    b.prepare_state_slots()
    assert b._runtime[0].ssm_cache[0].sum() == 0


def test_prefix_budget_rejects_oversized_snapshot(context):
    b = backend_with_state()
    b.prefix_state_budget_bytes = 1
    batch = Batch([request(256)], "prefill")
    b.capture_prefix_states(batch)
    assert batch.prefix_states == {}


def test_prefill_and_decode_share_one_gdn(monkeypatch):
    import minisgl.attention as attention
    from minisgl.attention.base import HybridBackend

    class Factories(dict):
        def assert_supported(self, names):
            assert all(n in self for n in names)

    monkeypatch.setattr(
        attention,
        "SUPPORTED_ATTENTION_BACKENDS",
        Factories(fa=lambda c: object(), fi=lambda c: object()),
    )
    b = attention.create_attention_backend("fa,fi", SimpleNamespace(has_linear_layers=True))
    assert isinstance(b, HybridLinearBackend)
    assert isinstance(b.full_backend, HybridBackend)
    assert not hasattr(b.full_backend.prefill_backend, "gdn_backend")
    assert not hasattr(b.full_backend.decode_backend, "gdn_backend")


def test_sparse_kv_layer_mapping(monkeypatch):
    import minisgl.kvcache.mha_pool as module

    monkeypatch.setattr(module, "get_tp_info", lambda: SimpleNamespace(size=1))
    pool = MHAKVCache(2, 2, 4, 3, 1, torch.float32, torch.device("cpu"), [3, 7])
    assert pool._kv_buffer.shape == (2, 2, 3, 1, 2, 4)
    pool.k_cache(3).fill_(3)
    pool.k_cache(7).fill_(7)
    assert torch.all(pool.k_cache(3) == 3)
    with pytest.raises(KeyError):
        pool.k_cache(0)


def test_4b_memory_layout_and_admission():
    config = SimpleNamespace(
        full_attention_layer_ids=list(range(3, 32, 4)),
        num_layers=32,
        num_kv_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    layout = HybridMemoryLayout.from_model(config)
    assert layout.kv_bytes_per_token == 32768
    assert layout.state_bytes_per_slot == 48 * 2**20 + 1179648
    args = dict(
        available_bytes=1 << 30,
        page_size=256,
        layout=layout,
        slots=5,
        workspace_bytes=128 << 20,
        snapshot_bytes=0,
    )
    pages = plan_kv_pages(**args)
    assert (pages + 1) * 256 * layout.kv_bytes_per_token + 5 * layout.state_bytes_per_slot + (
        128 << 20
    ) <= 1 << 30
    with pytest.raises(ValueError):
        plan_kv_pages(**{**args, "slots": 257})
    with pytest.raises(ValueError):
        plan_kv_pages(**args, override=pages + 1)
