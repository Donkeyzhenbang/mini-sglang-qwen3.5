"""CPU-only arithmetic, explicitly not a measured GPU benchmark."""

import argparse
import json
from pathlib import Path

from minisgl.models.config import ModelConfig
from minisgl.runtime.memory import HybridMemoryLayout

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--config", required=True)
p.add_argument("--slots", type=int, default=2)
p.add_argument("--context", type=int, default=4096)
p.add_argument("--parameters", type=int, default=0, help="Optional estimated parameter count")
p.add_argument("--weight-bits", type=int, choices=[4, 8, 16, 32], default=16)
args = p.parse_args()
c = ModelConfig.from_hf(json.loads(Path(args.config).read_text()))
layout = HybridMemoryLayout.from_model(c)
print(
    json.dumps(
        dict(
            measured=False,
            kind="static_memory_estimate",
            kv_bytes_per_token=layout.kv_bytes_per_token,
            state_bytes_per_slot=layout.state_bytes_per_slot,
            state_pool_bytes=args.slots * layout.state_bytes_per_slot,
            kv_bytes_for_context=args.context * layout.kv_bytes_per_token,
            weight_payload_lower_bound_bytes=(args.parameters * args.weight_bits + 7) // 8,
            excludes=[
                "quantization_scales_and_zeros",
                "draft",
                "features",
                "workspace",
                "allocator_overhead",
            ],
            int4_loader_implemented=False,
        ),
        indent=2,
    )
)
