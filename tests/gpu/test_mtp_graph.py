import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("steps", [1, 3])
@torch.inference_mode()
def test_mtp_graph_confirmed_kv_reset_reorder_fallback_boundary(steps):
    from minisgl.speculative.mtp import Qwen3_5MTPDraft, propose_mtp_batch
    from minisgl.speculative.mtp_graph import MTPGraphPool

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.manual_seed(908)
        config = dict(
            hidden_size=64, intermediate_size=128, head_dim=32,
            num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-6,
            mtp_num_hidden_layers=1,
            rope_parameters=dict(rope_type="default", rope_theta=1e7,
                                 partial_rotary_factor=0.5),
        )
        model = Qwen3_5MTPDraft(config, max_steps=steps).cuda().bfloat16().eval()
        attn = model.layer.self_attn
        inv = 1.0 / (attn.theta ** (torch.arange(0, 16, 2, device="cuda").float() / 16))
        angles = torch.arange(512, device="cuda").float()[:, None] * inv
        attn.rope_cache = torch.cat([angles.cos(), angles.sin()], -1)
        weights = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        actual = [model.fork_context() for _ in range(4)]
        reference = [model.fork_context() for _ in range(4)]
        slots = [3, 0, 2, 1]
        pool = MTPGraphPool(model, weights, weights, 4, 512, stream)

        def step(indices, counts, blocks=None):
            rows, oracle = [], []
            for i, count in zip(indices, counts):
                hidden = torch.randn(1, count, 64, device="cuda", dtype=torch.bfloat16)
                tokens = [(i * 17 + n) % 128 for n in range(count)]
                length = actual[i].context_length + count
                block = blocks[len(rows)] if blocks else steps + 1
                rows.append((actual[i], hidden, tokens, block, length))
                oracle.append((reference[i], hidden, tokens, block, length))
            expected = propose_mtp_batch(oracle, weights, weights)
            proposed = pool.propose(rows, [slots[i] for i in indices])
            if proposed is None:
                proposed = propose_mtp_batch(rows, weights, weights)
            assert proposed == expected
            for a, b in zip(actual, reference):
                assert a.context_length == b.context_length
                if a.cached_k is not None:
                    assert a.cached_k.shape[-2] == a.context_length
                    torch.testing.assert_close(a.cached_k, b.cached_k, atol=0.016, rtol=0.008)
                    torch.testing.assert_close(a.cached_v, b.cached_v, atol=0.016, rtol=0.008)

        step([0, 1, 2, 3], [17, 18, 19, 20])  # eager prefill -> graph import
        step([0, 1, 2, 3], [1, 3, 2, 4])
        step([0, 1, 2, 3], [4, 1, 3, 2])  # replay with different counts
        step([2, 0], [2, 1])
        actual[0].reset()
        reference[0].reset()
        step([0, 3], [3, 2])
        if steps == 3:
            step([0, 3], [1, 1], [2, 4])  # graph -> eager -> graph
        step([0, 3], [2, 3])
        desired = [505, 491, 503, 496]
        step([0, 1, 2, 3], [n - a.context_length for n, a in zip(desired, actual)])
        step([0, 1, 2, 3], [3, 16, 4, 10])  # padded RoPE positions exceed capacity
        assert pool.replays >= 6
        assert pool.fallbacks >= 2
        with pytest.raises(ValueError, match="distinct"):
            row = (actual[0], torch.randn(1, 1, 64, device="cuda"), [1], 2, 509)
            pool.propose([row, row], [3, 3])
    torch.cuda.current_stream().wait_stream(stream)
