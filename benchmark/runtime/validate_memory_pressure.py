"""Exercise real GPU cache release and block fallback with an injected budget.

This lowers the runtime's software budget; it does not fill a 24GB GPU or claim
to demonstrate capacity for a 27B quantized model.
"""

import argparse
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    from minisgl.distributed import DistributedInfo
    from minisgl.engine import Engine, EngineConfig
    from minisgl.runtime.hybrid_cache import HybridPrefixCache
    from minisgl.speculative.draft import DFlashDraft
    from minisgl.speculative.target import MiniSGLTarget

    config = EngineConfig(
        model_path=args.model,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        max_running_req=1,
        attention_backend="fi",
        cuda_graph_max_bs=0,
        page_size=256,
        num_page_override=4,
        max_seq_len_override=1024,
        prefix_state_budget_bytes=0,
        external_memory_bytes=2 << 30,
        gdn_extend_backend="packed",
    )
    engine = Engine(config)
    try:
        draft = DFlashDraft.from_directory(args.draft, engine.device, torch.bfloat16)
        cache = HybridPrefixCache(128 << 20, 256 << 20)
        target = MiniSGLTarget(engine, capture_layer_ids=draft.target_layer_ids, cache=cache)
        target.prefill([123, 456, 789, 234] * 16)
        blocks = (1, 2, 4, 8, 16)
        before = target.feasible_blocks(target.length, blocks)
        released = cache.used("gpu")
        assert released > 0
        allocated_before = torch.cuda.memory_allocated(engine.device)
        target.budget_bytes = allocated_before - released + target.safety_bytes + (4 << 20)
        pressured = target.feasible_blocks(target.length, blocks)
        allocated_after = torch.cuda.memory_allocated(engine.device)
        assert pressured == [1], pressured
        assert cache.used("gpu") == 0 and cache.used("cpu") == released
        assert allocated_after < allocated_before
        target.budget_bytes = 24 << 30
        recovered = target.feasible_blocks(target.length, blocks)
        assert recovered == list(blocks)
        output = dict(
            measured=True,
            pressure_source="injected_software_budget",
            before=before,
            under_pressure=pressured,
            recovered=recovered,
            allocated_before=allocated_before,
            allocated_after=allocated_after,
            cache_bytes_released=released,
            events=target.memory_events,
            cache=cache.stats,
        )
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(json.dumps(output))
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
