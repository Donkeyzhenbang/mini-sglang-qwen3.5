import copy
import json
import struct

import pytest
from minisgl.runtime.analyze import check_tokens
from minisgl.runtime.benchmark import checkpoint_bytes


def test_checkpoint_header_inspection_without_tensor_load(tmp_path):
    header = json.dumps(
        {"layer.weight": {"dtype": "BF16", "shape": [4, 8], "data_offsets": [0, 64]}}
    ).encode()
    (tmp_path / "weights.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + bytes(64)
    )
    assert checkpoint_bytes(tmp_path) == 64


def test_comparison_rejects_mismatched_or_unmeasured_runs():
    base = dict(
        measured=True,
        workload_sha256="abc",
        target_config_sha256="def",
        gpu="test",
        requests=[dict(token_ids=[1, 2, 3])],
    )
    assert check_tokens([base, copy.deepcopy(base)])
    for key, value in [
        ("measured", False),
        ("workload_sha256", "different"),
        ("requests", [dict(token_ids=[1, 3])]),
    ]:
        other = copy.deepcopy(base)
        other[key] = value
        with pytest.raises(ValueError):
            check_tokens([base, other])


def test_pinned_4b_configs_and_meta_draft():
    from pathlib import Path

    import torch
    from minisgl.models.config import ModelConfig
    from minisgl.runtime.benchmark import validate_model_pair
    from minisgl.speculative.draft import DFlashDraft

    root = Path(__file__).resolve().parents[2] / "benchmark/runtime/configs"
    target = ModelConfig.from_hf(json.loads((root / "Qwen3.5-4B/config.json").read_text()))
    draft = json.loads((root / "Qwen3.5-4B-DFlash/config.json").read_text())
    validate_model_pair(target, draft, 16)
    assert len(target.full_attention_layer_ids) == 8
    with torch.device("meta"):
        model = DFlashDraft(draft)
    assert model.fc.weight.shape == (2560, 8 * 2560)
    assert all(p.device.type == "meta" for p in model.parameters())
    bad = copy.deepcopy(draft)
    bad["hidden_size"] = 1024
    with pytest.raises(ValueError):
        validate_model_pair(target, bad, 16)
