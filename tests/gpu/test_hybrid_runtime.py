import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


def test_packed_extend_matches_recurrent_oracle():
    from minisgl.kernel.triton.gdn_extend import packed_extend

    torch.manual_seed(8)
    h, hv, k, v = 1, 2, 16, 16
    lengths, slots = [7, 2, 4], [2, 0, 1]
    count = sum(lengths)
    qkv = torch.randn(count, 2 * h * k + hv * v, device="cuda")
    a, b = (torch.randn(count, hv, device="cuda") for _ in range(2))
    alog, dt = (torch.randn(hv, device="cuda") for _ in range(2))
    initial = torch.randn(3, hv, v, k, device="cuda")
    actual_state = initial.clone()
    actual = packed_extend(
        qkv,
        a,
        b,
        alog,
        dt,
        actual_state,
        torch.tensor(slots, dtype=torch.int32, device="cuda"),
        torch.tensor([0, 7, 9, 13], dtype=torch.int32, device="cuda"),
        h,
        hv,
        k,
        v,
    )
    expected_state = initial.clone()
    expected, offset = [], 0
    for slot, length in zip(slots, lengths):
        state = expected_state[slot]
        for i in range(offset, offset + length):
            q, key, value = qkv[i].split([h * k, h * k, hv * v])
            q, key = (
                q.view(h, k).repeat_interleave(hv // h, 0),
                key.view(h, k).repeat_interleave(hv // h, 0),
            )
            q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) / k**0.5
            key = key / torch.sqrt((key * key).sum(-1, keepdim=True) + 1e-6)
            decay = (-alog.exp() * F.softplus(a[i] + dt)).exp()
            state = state * decay[:, None, None]
            delta = (value.view(hv, v) - torch.einsum("hvk,hk->hv", state, key)) * b[i].sigmoid()[
                :, None
            ]
            state = state + delta[:, :, None] * key[:, None, :]
            expected.append(torch.einsum("hvk,hk->hv", state, q))
        expected_state[slot] = state
        offset += length
    torch.testing.assert_close(actual, torch.stack(expected), rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual_state, expected_state, rtol=2e-4, atol=2e-4)


def test_gpu_eviction_offloads_complete_bundle():
    from minisgl.runtime.hybrid_cache import HybridPrefixCache

    cache = HybridPrefixCache(32, 64)
    payload = {
        "kv": torch.arange(4, dtype=torch.float32, device="cuda"),
        "ssm": torch.ones(4, device="cuda"),
    }
    cache.put([1], payload, 100)
    cache.put([2], payload, 100)
    entry = cache.lookup([1, 9])
    assert entry.tier == "cpu"
    torch.testing.assert_close(entry.tensors["kv"].cuda(), payload["kv"])
    assert cache.stats["offloads"] == 1
    cache.resize_gpu_budget(0)
    assert cache.used("gpu") == 0 and cache.used("cpu") == 64
