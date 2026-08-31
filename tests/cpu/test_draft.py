import torch
from minisgl.speculative.draft import DFlashDraft, visibility


def config():
    return dict(
        architectures=["DFlashDraftModel"],
        model_type="qwen3",
        dflash_config=dict(block_size=4, mask_token_id=30, target_layer_ids=[0, 2]),
        hidden_size=8,
        intermediate_size=12,
        head_dim=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=2,
        rms_norm_eps=1e-6,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=3,
        rope_parameters=dict(rope_type="default", rope_theta=10000),
    )


def test_layer_masks_match_window_and_noncausal_contract():
    q, k = torch.tensor([4, 5]), torch.arange(6)
    m = visibility(q, k, True, 3)[0, 0]
    assert m.tolist() == [
        [False, False, True, True, True, False],
        [False, False, False, True, True, True],
    ]
    assert visibility(q, k, False, None).all()


@torch.inference_mode()
def test_incremental_context_matches_uncached_draft():
    torch.manual_seed(7)
    m = DFlashDraft(config()).eval()
    features = torch.randn(1, 7, 16)
    noise = torch.randn(1, 4, 8)
    expected = m(features, noise, 7)
    m.reset()
    m(features[:, :4], torch.randn(1, 2, 8), 4)
    actual = m(features[:, 4:], noise, 7)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert m.layers[0].self_attn.cached_k.shape[-2] == 3
    assert m.layers[1].self_attn.cached_k.shape[-2] == 7


@torch.inference_mode()
def test_proposal_uses_anchor_and_returns_block():
    torch.manual_seed(8)
    model = DFlashDraft(config()).eval()
    embedding = torch.randn(32, 8)
    block = model.propose(torch.randn(1, 5, 16), 7, 4, 5, embedding, embedding)
    assert len(block) == 4 and block[0] == 7
    assert all(0 <= token < 32 for token in block)


def test_native_checkpoint_loading(tmp_path):
    import json

    from safetensors.torch import save_file

    m = DFlashDraft(config()).eval()
    (tmp_path / "config.json").write_text(json.dumps(config()))
    save_file(m.state_dict(), tmp_path / "model.safetensors")
    restored = DFlashDraft.from_directory(tmp_path, "cpu", torch.float32)
    for name, value in m.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


@torch.inference_mode()
def test_request_context_forks_share_weights_but_not_kv():
    torch.manual_seed(41)
    original = DFlashDraft(config()).eval()
    first, second = original.fork_context(), original.fork_context()
    assert (
        first.fc.weight.data_ptr() == second.fc.weight.data_ptr() == original.fc.weight.data_ptr()
    )
    a, b, noise = torch.randn(1, 5, 16), torch.randn(1, 3, 16), torch.randn(1, 4, 8)
    first(a, noise, 5)
    before = first.layers[1].self_attn.cached_k.clone()
    second(b, noise, 3)
    assert first.context_length == 5 and second.context_length == 3 and original.context_length == 0
    torch.testing.assert_close(first.layers[1].self_attn.cached_k, before, rtol=0, atol=0)
    second.reset()
    assert first.layers[1].self_attn.cached_k is not None


@torch.inference_mode()
def test_ragged_batched_draft_matches_independent_requests_and_cache_reuse():
    from minisgl.speculative.batch_draft import propose_batch

    torch.manual_seed(73)
    model = DFlashDraft(config()).eval()
    batched = [model.fork_context() for _ in range(4)]
    serial = [model.fork_context() for _ in range(4)]
    embedding = torch.randn(32, 8)
    # Ragged prompts, differing blocks, compaction, noncontiguous slot order,
    # accumulated confirmed context after a block=1 fallback, and slot reuse.
    for step, slots in enumerate([(0, 1, 2, 3), (3, 1, 0), (1,), (2, 0, 3, 1)]):
        if step == 3:
            batched[2].reset()
            serial[2].reset()
        rows, expected = [], []
        inactive = {
            i: (d.context_length, d.layers[1].self_attn.cached_k.clone())
            for i, d in enumerate(batched)
            if i not in slots and d.context_length
        }
        for i in slots:
            n = (i + step) % 5 + 1
            features = torch.randn(1, n, 16)
            anchor, block = 3 + i, 2 + i % 3
            length = serial[i].context_length + n
            rows.append((batched[i], features, anchor, block, length))
            expected.append(
                serial[i].propose(features, anchor, block, length, embedding, embedding)
            )
        assert propose_batch(rows, embedding, embedding) == expected
        for i in slots:
            assert batched[i].context_length == serial[i].context_length
            for a, b in zip(batched[i].layers, serial[i].layers):
                for name in ["cached_k", "cached_v"]:
                    torch.testing.assert_close(
                        getattr(a.self_attn, name), getattr(b.self_attn, name), rtol=2e-5, atol=2e-5
                    )
        for i, (length, keys) in inactive.items():
            assert batched[i].context_length == length
            torch.testing.assert_close(
                batched[i].layers[1].self_attn.cached_k, keys, rtol=0, atol=0
            )


def test_batched_draft_rejects_duplicate_contexts_and_unshared_weights():
    import pytest
    from minisgl.speculative.batch_draft import propose_batch

    model = DFlashDraft(config()).eval()
    row = (model, torch.randn(1, 2, 16), 4, 4, 2)
    weights = torch.randn(32, 8)
    with pytest.raises(ValueError, match="distinct"):
        propose_batch([row, row], weights, weights)
    other = DFlashDraft(config()).eval()
    with pytest.raises(ValueError, match="share model weights"):
        propose_batch([row, (other, *row[1:])], weights, weights)
