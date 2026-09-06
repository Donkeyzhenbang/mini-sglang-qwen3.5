"""Guarded Qwen3.5-4B draft-config backport for the unmerged SGLang PR #19952.

This modifies only an explicitly supplied, isolated source checkout, not an
installed package or the MiniSGLang target. Run without --apply to inspect.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ORIGINAL_SHA = "6dfc4b146c2cab2831cb017df21eeca5919e5ad296b99e99072e6b68df6841b5"
PATCHED_SHA = "a23935c81be0e3d8557c7e0c7d9ee6de8bc7d5fcc67b180648f5ca71ae003609"
SOURCE_REVISION = "5106aa523aee7a77471fdb50e0cb1e8b21da9427"
HELPER = '''
def resolve_dflash_attention_config(config, layer_id):
    # Recent checkpoints serialize RoPE under rope_parameters. Older HF
    # Qwen3Config otherwise supplies a legacy default rope_theta silently.
    parameters = getattr(config, "rope_parameters", None) or {}
    if parameters and parameters.get("rope_type", "default") != "default":
        raise ValueError("This PR backport only validates default rope_parameters")
    theta = float(parameters.get("rope_theta", getattr(config, "rope_theta", 1000000)))
    layer_types = getattr(config, "layer_types", None)
    layer_type = layer_types[layer_id] if layer_types else "full_attention"
    if layer_type not in ("sliding_attention", "full_attention"):
        raise ValueError(f"Unknown DFlash layer type: {layer_type}")
    configured_causal = getattr(config, "is_causal", None)
    causal = layer_type == "sliding_attention" if configured_causal is None else bool(configured_causal)
    window_left = -1
    if layer_type == "sliding_attention":
        window = int(config.sliding_window)
        if window < 2 or not causal:
            raise ValueError("This PR backport supports causal sliding windows >= 2")
        # Reference mask: query_position - key_position < window.
        # FlashInfer's window_left counts previous positions, excluding self.
        window_left = window - 1
    return theta, causal, window_left

'''


def patched_source(source):
    if "def resolve_dflash_attention_config(" in source:
        if hashlib.sha256(source.encode()).hexdigest() != PATCHED_SHA:
            raise ValueError("Unexpected modified source; refusing an ambiguous reapply")
        return source
    if hashlib.sha256(source.encode()).hexdigest() != ORIGINAL_SHA:
        raise ValueError("Unexpected dflash.py source; refusing to patch a different revision")
    source = source.replace("class DFlashAttention(nn.Module):", HELPER + "class DFlashAttention(nn.Module):", 1)
    old = '        rope_theta = float(getattr(config, "rope_theta", 1000000))'
    assert source.count(old) == 1
    source = source.replace(old, '        rope_theta, is_causal, window_left = resolve_dflash_attention_config(config, layer_id)', 1)
    source = source.replace("        # DFlash uses non-causal attention over the draft block.",
                            "        # Preserve checkpoint per-layer masking; only full layers default to non-causal.", 1)
    old = "            attn_type=AttentionType.ENCODER_ONLY,"
    assert source.count(old) == 1
    source = source.replace(old, "            sliding_window_size=window_left,\n            attn_type=AttentionType.DECODER if is_causal else AttentionType.ENCODER_ONLY,", 1)
    return source


def validate(source, config):
    tree = ast.parse(source)
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "resolve_dflash_attention_config")
    namespace = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "<actual patched helper>", "exec"), namespace)
    resolve = namespace[helper.name]
    # Simulate legacy HF's injected default, which must not override serialized theta.
    loaded = dict(config)
    loaded["rope_theta"] = 10000.0
    got = [resolve(SimpleNamespace(**loaded), i) for i in range(6)]
    assert got == [(10000000.0, True, 4095)] * 5 + [(10000000.0, False, -1)]
    assert resolve(SimpleNamespace(rope_theta=12345.0), 0) == (12345.0, False, -1)
    # Independent visibility oracle at a sliding-window boundary, not merely
    # an equality check against the implementation's returned integer.
    q = 4100
    reference = {k for k in range(q + 4) if k <= q and q - k < 4096}
    flashinfer_semantics = {k for k in range(q + 4) if q - got[0][2] <= k <= q}
    assert reference == flashinfer_semantics and min(reference) == 5
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="Isolated SGLang PR source repository root")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    path = args.source / "python/sglang/srt/models/dflash.py"
    before = path.read_text()
    after = patched_source(before)
    layers = validate(after, json.loads(args.config.read_text()))
    if args.apply and before != after:
        backup = path.with_suffix(".py.pr19952-original")
        if backup.exists():
            raise ValueError("Refusing to overwrite existing backup")
        backup.write_text(before)
        path.write_text(after)
    print(json.dumps(dict(source_revision=SOURCE_REVISION, applied=args.apply,
        changed=before != after, original_sha256=ORIGINAL_SHA,
        patched_sha256=hashlib.sha256(after.encode()).hexdigest(), layers=layers,
        target_or_acceptance_logic_modified=False, gpu_validation_pending=True), indent=2))


if __name__ == "__main__":
    main()
