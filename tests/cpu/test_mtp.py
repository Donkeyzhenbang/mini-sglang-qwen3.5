import json

import torch
from minisgl.speculative.mtp import (
    GemmaRMSNorm,
    Qwen3_5MTPDraft,
    propose_mtp_batch,
)


def config():
    return {
        "hidden_size": 8,
        "intermediate_size": 12,
        "head_dim": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
        "vocab_size": 32,
        "mtp_num_hidden_layers": 1,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
        },
    }


def test_gemma_norm_uses_checkpoint_weight_offset():
    norm = GemmaRMSNorm(4, 1e-6)
    norm.weight.data.fill_(0.5)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * 1.5
    torch.testing.assert_close(norm(x), expected)


@torch.inference_mode()
def test_mtp3_batched_matches_serial_and_commits_confirmed_kv_only():
    torch.manual_seed(17)
    base = Qwen3_5MTPDraft(config(), max_steps=3).eval()
    batched = [base.fork_context() for _ in range(3)]
    serial = [base.fork_context() for _ in range(3)]
    embedding = torch.randn(32, 8)
    head = torch.randn(32, 8)

    for iteration, slots in enumerate([(0, 1, 2), (2, 0), (1, 2, 0)]):
        rows, expected = [], []
        for slot in slots:
            count = 1 + (slot + iteration) % 4
            hidden = torch.randn(count, 8)
            tokens = [2 + ((slot * 5 + iteration + i) % 20) for i in range(count)]
            target_length = serial[slot].context_length + count
            block = 4 if (slot + iteration) % 2 == 0 else 2
            rows.append(
                (batched[slot], hidden, tokens, block, target_length)
            )
            expected.append(
                serial[slot].propose(
                    hidden,
                    tokens,
                    block,
                    target_length,
                    embedding,
                    head,
                )
            )

        actual = propose_mtp_batch(rows, embedding, head)
        assert actual == expected
        for row, slot in zip(rows, slots):
            draft, _, _, block, target_length = row
            assert len(actual[slots.index(slot)]) == block
            assert draft.context_length == target_length
            assert draft.cached_k.shape[-2] == target_length
            torch.testing.assert_close(
                draft.cached_k, serial[slot].cached_k, rtol=2e-5, atol=2e-5
            )
            torch.testing.assert_close(
                draft.cached_v, serial[slot].cached_v, rtol=2e-5, atol=2e-5
            )


@torch.inference_mode()
def test_incremental_confirmed_context_matches_full_rebuild():
    torch.manual_seed(29)
    base = Qwen3_5MTPDraft(config(), max_steps=3).eval()
    incremental, rebuilt = base.fork_context(), base.fork_context()
    embedding = torch.randn(32, 8)
    head = torch.randn(32, 8)
    hidden = torch.randn(7, 8)
    tokens = [3, 4, 5, 6, 7, 8, 9]

    incremental.propose(hidden[:3], tokens[:3], 4, 3, embedding, head)
    actual = incremental.propose(hidden[3:], tokens[3:], 4, 7, embedding, head)
    expected = rebuilt.propose(hidden, tokens, 4, 7, embedding, head)

    assert actual == expected
    torch.testing.assert_close(
        incremental.cached_k, rebuilt.cached_k, rtol=2e-5, atol=2e-5
    )
    torch.testing.assert_close(
        incremental.cached_v, rebuilt.cached_v, rtol=2e-5, atol=2e-5
    )


def test_official_embedded_checkpoint_loading(tmp_path):
    from safetensors.torch import save_file

    torch.manual_seed(41)
    model = Qwen3_5MTPDraft(config(), max_steps=3).eval()
    state = dict(model.state_dict())

    attn = "layer.self_attn."
    qkv = state.pop(attn + "qkv_proj.weight")
    q, k, v = qkv.split(
        [
            model.layer.self_attn.q_size,
            model.layer.self_attn.kv_size,
            model.layer.self_attn.kv_size,
        ],
        dim=0,
    )
    state[attn + "q_proj.weight"] = q
    state[attn + "k_proj.weight"] = k
    state[attn + "v_proj.weight"] = v

    mlp = "layer.mlp."
    gate, up = state.pop(mlp + "gate_up_proj.weight").chunk(2, dim=0)
    state[mlp + "gate_proj.weight"] = gate
    state[mlp + "up_proj.weight"] = up

    checkpoint = {}
    for name, value in state.items():
        name = name.replace("layer.", "layers.0.", 1)
        checkpoint["mtp." + name] = value
    save_file(checkpoint, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": config()})
    )

    restored = Qwen3_5MTPDraft.from_directory(
        tmp_path, "cpu", torch.float32, max_steps=3
    )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)
    first, second = restored.fork_context(), restored.fork_context()
    assert first.fc.weight.data_ptr() == second.fc.weight.data_ptr()
    assert first.cached_k is None and second.cached_k is None


def test_mtp_rejects_duplicate_request_contexts():
    import pytest

    model = Qwen3_5MTPDraft(config(), max_steps=3).eval()
    hidden = torch.randn(2, 8)
    row = (model, hidden, [3, 4], 4, 2)
    weights = torch.randn(32, 8)
    with pytest.raises(ValueError, match="distinct"):
        propose_mtp_batch([row, row], weights, weights)
