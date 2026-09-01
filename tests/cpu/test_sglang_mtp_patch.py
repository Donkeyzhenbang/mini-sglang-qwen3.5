import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[2] / "benchmark/runtime/patch_sglang_059_mrope.py"
    spec = importlib.util.spec_from_file_location("patch_sglang_059_mrope", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_patch_is_narrow_and_idempotent():
    source = """\
    def _compute_mrope_positions(self):
        if self.forward_mode.is_decode():
            pass
        elif self.forward_mode.is_extend():
            pass

    def _pad_tensor_to_size(self):
        if self.forward_mode.is_extend():
            pass
"""
    module = _module()
    updated, changed = module.patch_text(source)
    assert changed
    assert updated.count("include_draft_extend_v2=True") == 1
    assert "if self.forward_mode.is_extend():" in updated.split("def _pad_tensor_to_size", 1)[1]
    repeated, changed = module.patch_text(updated)
    assert not changed
    assert repeated == updated


def test_hybrid_patch_skips_gdn_metadata_for_draft_v2():
    source = """\
    def _forward_metadata(self):
        if forward_batch.forward_mode.is_decode_or_idle():
            pass
        elif forward_batch.forward_mode.is_extend():
            if forward_batch.forward_mode.is_target_verify():
                pass
        else:
            raise ValueError

    def init_forward_metadata(self):
        pass
"""
    module = _module()
    updated, changed = module.patch_hybrid_text(source)
    assert changed
    assert "is_extend(include_draft_extend_v2=True)" in updated
    assert "if forward_batch.forward_mode.is_draft_extend_v2():" in updated
    assert "query_start_loc = None" in updated
    repeated, changed = module.patch_hybrid_text(updated)
    assert not changed
    assert repeated == updated
