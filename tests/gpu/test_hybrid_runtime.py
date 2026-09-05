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


@pytest.mark.parametrize("hv", [16, 32])
def test_bf16_packed_extend_matches_stepwise_decode(hv):
    from minisgl.kernel.triton.gdn_decode import packed_decode
    from minisgl.kernel.triton.gdn_extend import packed_extend

    torch.manual_seed(17)
    h, k, v, count = 16, 128, 128, 17
    qkv = torch.randn(count, 2 * h * k + hv * v, device="cuda", dtype=torch.bfloat16)
    a, b = (torch.randn(count, hv, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    alog, dt = (torch.randn(hv, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    initial = torch.randn(1, hv, v, k, device="cuda")
    slots = torch.tensor([0], device="cuda", dtype=torch.int32)
    state = initial.clone()
    actual = packed_extend(
        qkv,
        a,
        b,
        alog,
        dt,
        state,
        slots,
        torch.tensor([0, count], device="cuda", dtype=torch.int32),
        h,
        hv,
        k,
        v,
    )
    expected_state = initial.clone()
    expected = torch.cat(
        [
            packed_decode(
                qkv[i : i + 1],
                a[i : i + 1],
                b[i : i + 1],
                alog,
                dt,
                expected_state,
                slots,
                h,
                hv,
                k,
                v,
                k**-0.5,
            )
            for i in range(count)
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


def test_bf16_single_token_conv_matches_full_convolution():
    from minisgl.attention.gdn import GDNAttnBackend, _LayerRuntime

    torch.manual_seed(11)
    x = torch.randn(1, 8192, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(8192, 1, 4, device="cuda", dtype=torch.bfloat16)
    initial = torch.randn(1, 8192, 3, device="cuda", dtype=torch.bfloat16)
    rt = _LayerRuntime(initial.clone(), torch.empty(0))
    expected = F.silu(F.conv1d(torch.cat([initial, x.unsqueeze(-1)], -1), w, groups=8192))
    actual = GDNAttnBackend()._apply_conv(x, 0, w, rt)
    torch.testing.assert_close(actual, expected.squeeze(-1), rtol=0, atol=0)
    torch.testing.assert_close(rt.conv_cache, torch.cat([initial[:, :, 1:], x.unsqueeze(-1)], -1))


@pytest.mark.parametrize("channels", [135, 8192])
def test_ragged_packed_convolution_matches_torch_and_preserves_inactive_slots(channels):
    from minisgl.attention.gdn import GDNAttnBackend, _LayerRuntime
    from minisgl.kernel.triton.conv_extend import packed_conv_extend

    torch.manual_seed(918)
    lengths, slots = [8, 1, 2, 16], [4, 0, 3, 1]
    x = torch.randn(sum(lengths), channels, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(channels, 1, 4, device="cuda", dtype=torch.bfloat16)
    initial = torch.randn(5, channels, 3, device="cuda", dtype=torch.bfloat16)
    expected_rt = _LayerRuntime(initial.clone(), torch.empty(0))
    expected, offset = [], 0
    cu = [0]
    for slot, length in zip(slots, lengths):
        expected.append(
            GDNAttnBackend()._apply_conv(x[offset : offset + length], slot, w, expected_rt)
        )
        offset += length
        cu.append(offset)
    actual_state = initial.clone()
    actual = packed_conv_extend(
        x,
        w,
        actual_state,
        torch.tensor(slots, device="cuda", dtype=torch.int32),
        torch.tensor(cu, device="cuda", dtype=torch.int32),
    )
    torch.testing.assert_close(actual, torch.cat(expected), rtol=0, atol=0)
    torch.testing.assert_close(actual_state, expected_rt.conv_cache, rtol=0, atol=0)
    torch.testing.assert_close(actual_state[2], initial[2], rtol=0, atol=0)


def test_decode_convolution_matches_bf16_prefill_rounding():
    from minisgl.attention.gdn import GDNAttnBackend, _LayerRuntime

    torch.manual_seed(118)
    x = torch.randn(4, 8192, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(8192, 1, 4, device="cuda", dtype=torch.bfloat16)
    initial = torch.randn(4, 8192, 3, device="cuda", dtype=torch.bfloat16)
    rt = _LayerRuntime(initial.clone(), torch.empty(0))
    expected = F.silu(F.conv1d(torch.cat([initial, x.unsqueeze(-1)], -1), w, groups=8192)).squeeze(
        -1
    )
    actual = GDNAttnBackend()._apply_conv_decode_batch(
        x, w, rt, torch.arange(4, device="cuda", dtype=torch.int32)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("lengths", [[1, 3], [4, 2], [0, 3]])
def test_selective_journal_replay_matches_compacted_prefixes(lengths):
    from minisgl.kernel.triton.conv_extend import packed_conv_extend
    from minisgl.kernel.triton.gdn_extend import packed_extend

    torch.manual_seed(113)
    h, hv, k, v = 2, 4, 128, 128
    d = 2 * h * k + hv * v
    x = torch.randn(16, d, device="cuda", dtype=torch.bfloat16)
    a, b = (torch.randn(16, hv, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    alog, dt = (torch.randn(hv, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    w = torch.randn(d, 1, 4, device="cuda", dtype=torch.bfloat16)
    conv = torch.randn(5, d, 3, device="cuda", dtype=torch.bfloat16)
    ssm = torch.randn(5, hv, v, k, device="cuda")
    ref_conv, ref_ssm = conv.clone(), ssm.clone()
    # Reordered requests, omitted full-accept requests, and a zero-length prefix.
    starts = [12, 4]
    ends = [start + n for start, n in zip(starts, lengths)]
    indices = [i for start, end in zip(starts, ends) for i in range(start, end)]
    meta = torch.tensor([[3, 1], starts, ends], device="cuda", dtype=torch.int32)
    slots, cu = (
        meta[0],
        torch.tensor([0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32),
    )
    y = packed_conv_extend(x, w, conv, slots, meta[1], end_offsets=meta[2])
    expected_y = packed_conv_extend(x[indices].contiguous(), w, ref_conv, slots, cu)
    actual = packed_extend(y, a, b, alog, dt, ssm, slots, meta[1], h, hv, k, v, end_offsets=meta[2])
    expected = packed_extend(
        expected_y,
        a[indices].contiguous(),
        b[indices].contiguous(),
        alog,
        dt,
        ref_ssm,
        slots,
        cu,
        h,
        hv,
        k,
        v,
    )
    torch.testing.assert_close(y[indices], expected_y, rtol=0, atol=0)
    torch.testing.assert_close(actual[indices], expected, rtol=0, atol=0)
    torch.testing.assert_close(conv, ref_conv, rtol=0, atol=0)
    torch.testing.assert_close(ssm, ref_ssm, rtol=0, atol=0)
