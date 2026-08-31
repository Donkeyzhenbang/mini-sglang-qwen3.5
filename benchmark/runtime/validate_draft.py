"""Compare the native draft to an explicitly supplied, reviewed upstream source.

No code is downloaded automatically. --reference-source executes that local
Python file, so use only a reviewed, pinned z-lab/dflash model.py.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draft", required=True)
    p.add_argument("--reference-source", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    from minisgl.speculative.draft import DFlashDraft
    from torch.nn.attention import SDPBackend, sdpa_kernel

    source = Path(args.reference_source)
    spec = importlib.util.spec_from_file_location("reviewed_dflash_reference", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for dtype in (torch.float32, torch.bfloat16):
            native = DFlashDraft.from_directory(args.draft, "cuda:0", dtype)
            reference = module.DFlashDraftModel.from_pretrained(
                args.draft, dtype=dtype, device_map={"": "cuda:0"}
            ).eval()
            torch.manual_seed(42)
            contexts = []
            length = 0
            for added, block in [(17, 8), (3, 2), (7, 16), (4, 4)]:
                context = torch.randn(1, added, native.fc.in_features, device="cuda", dtype=dtype)
                noise = torch.randn(
                    1, block, native.config["hidden_size"], device="cuda", dtype=dtype
                )
                contexts.append(context)
                length += added
                actual = native(context, noise, length)
                expected = reference(
                    position_ids=torch.arange(length + block, device="cuda")[None],
                    noise_embedding=noise,
                    target_hidden=torch.cat(contexts, dim=1),
                    use_cache=False,
                )
                delta = actual.float() - expected.float()
                result = dict(
                    dtype=str(dtype),
                    context=length,
                    block=block,
                    max_abs=delta.abs().max().item(),
                    relative_l2=(delta.norm() / expected.float().norm()).item(),
                )
                if dtype == torch.float32:
                    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
                results.append(result)
            del native, reference
            torch.cuda.empty_cache()
    output = dict(
        measured=True,
        reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        fp32_math_parity=True,
        results=results,
    )
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(json.dumps(output))


if __name__ == "__main__":
    main()
