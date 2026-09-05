import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


@pytest.mark.parametrize("width", [128, 2560])
def test_draft_fusions_match_bf16_reference(width):
    from minisgl.kernel.triton.draft_ops import rms_norm, silu_mul

    torch.manual_seed(921)
    x = torch.randn(7, width, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(width, device="cuda", dtype=torch.bfloat16)
    y = x.float()
    expected = (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * weight
    torch.testing.assert_close(rms_norm(x, weight, 1e-6), expected, atol=0.015625, rtol=0.008)
    packed = torch.randn(7, width * 2, device="cuda", dtype=torch.bfloat16)
    gate, up = packed.chunk(2, -1)
    torch.testing.assert_close(silu_mul(packed), F.silu(gate) * up, atol=0.015625, rtol=0.008)


def test_fused_rotary_preserves_bf16_rounding():
    from minisgl.kernel.triton.draft_ops import cached_rotary
    from minisgl.speculative.draft import rotary

    torch.manual_seed(292)
    x = torch.randn(3, 9, 4, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    positions = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [15, 16, 17, 18, 19, 20, 21, 22, 23],
            [40, 42, 44, 46, 48, 50, 52, 54, 56],
        ],
        device="cuda",
    )
    inv = 1.0 / (1e7 ** (torch.arange(0, 128, 2, device="cuda", dtype=torch.float32) / 128))
    angles = torch.arange(64, device="cuda", dtype=torch.float32)[:, None] * inv
    cache = torch.cat([angles.cos(), angles.sin()], -1).to(x.dtype)
    torch.testing.assert_close(
        cached_rotary(x, positions, cache), rotary(x, positions, 1e7), rtol=0, atol=0
    )


@pytest.fixture
def graph_stream():
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        yield stream
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize("use_rope_cache", [False, True])
@torch.inference_mode()
def test_dflash_graph_reordering_reset_fallback_and_capacity_boundary(use_rope_cache, graph_stream):
    from minisgl.speculative.batch_draft import propose_batch
    from minisgl.speculative.draft import DFlashDraft
    from minisgl.speculative.draft_graph import DFlashGraphPool

    torch.manual_seed(942)
    config = dict(
        architectures=["DFlashDraftModel"],
        model_type="qwen3",
        dflash_config=dict(block_size=16, mask_token_id=126, target_layer_ids=[0, 2]),
        hidden_size=32,
        intermediate_size=64,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        rms_norm_eps=1e-6,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=7,
        rope_parameters=dict(rope_type="default", rope_theta=10000),
    )
    model = DFlashDraft(config).cuda().eval()
    model.build_context_kv_fusion()
    if use_rope_cache:
        inv = 1.0 / (10000 ** (torch.arange(0, 16, 2, device="cuda").float() / 16))
        angles = torch.arange(512, device="cuda").float()[:, None] * inv
        table = torch.cat([angles.cos(), angles.sin()], -1)
        for layer in model.layers:
            layer.self_attn.rope_cache = table
    embedding = torch.randn(128, 32, device="cuda")
    actual = [model.fork_context() for _ in range(4)]
    reference = [model.fork_context() for _ in range(4)]
    slots = [3, 0, 2, 1]
    pool = DFlashGraphPool(model, embedding, embedding, 4, 512, torch.cuda.current_stream())

    def step(ids, counts, block):
        rows, oracle = [], []
        for i, n in zip(ids, counts):
            features = torch.randn(1, n, 64, device="cuda")
            length = actual[i].context_length + n
            anchor = (i * 17 + length) % 125
            rows.append((actual[i], features, anchor, block, length))
            oracle.append((reference[i], features, anchor, block, length))
        expected = propose_batch(oracle, embedding, embedding)
        proposed = pool.propose(rows, [slots[i] for i in ids])
        if proposed is None:
            proposed = propose_batch(rows, embedding, embedding)
        assert proposed == expected
        for a, b in zip(actual, reference):
            assert a.context_length == b.context_length
            for la, lb in zip(a.layers, b.layers):
                for name in ("cached_k", "cached_v"):
                    av, bv = getattr(la.self_attn, name), getattr(lb.self_attn, name)
                    if av is not None:
                        torch.testing.assert_close(av, bv, rtol=1e-4, atol=2e-5)

    step([0, 1, 2, 3], [17, 18, 19, 20], 4)
    step([0, 1, 2, 3], [1, 3, 2, 4], 4)
    step([2, 0], [2, 1], 4)
    actual[0].reset()
    reference[0].reset()
    step([0, 3], [3, 2], 4)
    wanted = [509, 494, 508, 503]
    step([0, 1, 2, 3], [n - a.context_length for n, a in zip(wanted, actual)], 2)
    # Padded context positions exceed capacity for some rows, real tokens do not.
    step([0, 1, 2, 3], [1, 16, 2, 7], 2)
    assert pool.replays == 4
    assert pool.fallbacks == 2


@pytest.mark.usefixtures("graph_stream")
@torch.inference_mode()
def test_journal_graph_dynamic_ranges_restore_capture_state():
    from types import SimpleNamespace

    from minisgl.kernel.triton.journal_graph import JournalReplayGraph, replay_journal

    torch.manual_seed(1987)
    h, hv, k, v = 2, 4, 128, 128
    d = h * k * 2 + hv * v
    runtime, reference, journal = {}, {}, {}
    for lid in range(3):
        conv = torch.randn(4, d, 3, device="cuda", dtype=torch.bfloat16)
        ssm = torch.randn(4, hv, v, k, device="cuda")
        runtime[lid] = SimpleNamespace(conv_cache=conv.clone(), ssm_cache=ssm.clone())
        reference[lid] = SimpleNamespace(conv_cache=conv.clone(), ssm_cache=ssm.clone())
        journal[lid] = SimpleNamespace(
            mixed_qkv=torch.randn(12, d, device="cuda", dtype=torch.bfloat16),
            a=torch.randn(12, hv, device="cuda", dtype=torch.bfloat16),
            b=torch.randn(12, hv, device="cuda", dtype=torch.bfloat16),
            conv_weight=torch.randn(d, 1, 4, device="cuda", dtype=torch.bfloat16),
            A_log=torch.randn(hv, device="cuda", dtype=torch.bfloat16),
            dt_bias=torch.randn(hv, device="cuda", dtype=torch.bfloat16),
            num_q_heads=h,
            num_v_heads=hv,
            head_k_dim=k,
            head_v_dim=v,
        )
    first = torch.tensor([[3, 0], [0, 4], [2, 7]], device="cuda", dtype=torch.int32)
    graph = JournalReplayGraph(journal, runtime, first, [3, 0])
    for lid in runtime:
        torch.testing.assert_close(
            runtime[lid].conv_cache, reference[lid].conv_cache, rtol=0, atol=0
        )
        torch.testing.assert_close(runtime[lid].ssm_cache, reference[lid].ssm_cache, rtol=0, atol=0)
    for values in [[[3, 0], [0, 4], [2, 7]], [[2, 3], [8, 0], [12, 1]]]:
        metadata = torch.tensor(values, device="cuda", dtype=torch.int32)
        for record in journal.values():
            record.mixed_qkv.add_(0.125)
            record.a.mul_(0.875)
        graph.replay(metadata)
        replay_journal(journal, reference, metadata)
        for lid in runtime:
            torch.testing.assert_close(
                runtime[lid].conv_cache, reference[lid].conv_cache, rtol=0, atol=0
            )
            torch.testing.assert_close(
                runtime[lid].ssm_cache, reference[lid].ssm_cache, rtol=0, atol=0
            )
