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
