"""Backport the Qwen3.5 fixes needed by SGLang 0.5.9 spec-v2.

SGLang 0.5.9 classifies DRAFT_EXTEND_V2 separately from ordinary extend.
Its old MRoPE helper forgot to opt into that mode, leaving Python lists in a
tensor concatenation. Newer SGLang handles this mode explicitly. This narrow
backport changes that conditional and teaches the hybrid backend that draft-v2
does not run GDN layers. Both changes match the newer SGLang control flow.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path


def patch_text(text: str) -> tuple[str, bool]:
    start = text.index("    def _compute_mrope_positions(")
    end = text.index("    def _pad_tensor_to_size(", start)
    block = text[start:end]
    old = "elif self.forward_mode.is_extend():"
    new = "elif self.forward_mode.is_extend(include_draft_extend_v2=True):"
    if new in block:
        return text, False
    if block.count(old) != 1:
        raise RuntimeError("Unexpected SGLang MRoPE helper; refusing an ambiguous patch")
    block = block.replace(old, new, 1)
    return text[:start] + block + text[end:], True


def patch_hybrid_text(text: str) -> tuple[str, bool]:
    start = text.index("    def _forward_metadata(")
    end = text.index("    def init_forward_metadata(", start)
    block = text[start:end]
    old = """\
        elif forward_batch.forward_mode.is_extend():
            if forward_batch.forward_mode.is_target_verify():
"""
    new = """\
        elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
            if forward_batch.forward_mode.is_draft_extend_v2():
                # Draft-v2 runs only full-attention layers in this MTP model.
                query_start_loc = None
            elif forward_batch.forward_mode.is_target_verify():
"""
    if new in block:
        return text, False
    if block.count(old) != 1:
        raise RuntimeError("Unexpected SGLang hybrid backend; refusing an ambiguous patch")
    block = block.replace(old, new, 1)
    return text[:start] + block + text[end:], True


def main() -> None:
    if importlib.metadata.version("sglang") != "0.5.9":
        raise RuntimeError("This backport is restricted to SGLang 0.5.9")
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        raise RuntimeError("Cannot locate the installed SGLang package")
    root = Path(spec.origin).parent / "srt"
    patches = [
        (root / "model_executor/forward_batch_info.py", patch_text, "text MRoPE"),
        (
            root / "layers/attention/hybrid_linear_attn_backend.py",
            patch_hybrid_text,
            "hybrid metadata",
        ),
    ]
    for path, transform, label in patches:
        updated, changed = transform(path.read_text())
        if changed:
            path.write_text(updated)
            print(f"Patched SGLang 0.5.9 spec-v2 {label}: {path}")
        else:
            print(f"SGLang 0.5.9 spec-v2 {label} backport already present: {path}")


if __name__ == "__main__":
    main()
