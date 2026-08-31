"""Real-model GPU checks for prefix restoration and speculative rollback.

Compare identical segmentation for exact transaction checks. Report full-vs-split
prefill drift separately: different GEMM shapes need not be bitwise identical.
"""

import argparse
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--workload", required=True)
    p.add_argument("--row", type=int, default=9)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    from minisgl.distributed import DistributedInfo
    from minisgl.engine import Engine, EngineConfig
    from minisgl.runtime.hybrid_cache import HybridPrefixCache
    from minisgl.speculative.target import MiniSGLTarget

    ids = json.loads(Path(args.workload).read_text().splitlines()[args.row])["input_ids"]
    config = EngineConfig(
        model_path=args.model,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        max_running_req=1,
        attention_backend="fi",
        cuda_graph_max_bs=0,
        page_size=256,
        num_page_override=8,
        max_seq_len_override=2048,
        prefix_state_budget_bytes=0,
        gdn_extend_backend="packed",
    )
    engine = Engine(config)
    taps = tuple(range(1, config.model_config.num_layers - 1, 4))
    target = MiniSGLTarget(engine, capture_layer_ids=taps)
    report = {
        "measured": True,
        "model": args.model,
        "gpu": torch.cuda.get_device_name(),
        "checks": {},
    }

    def snapshot(logits, features):
        target.synchronize()
        return {k: v.detach().clone() for k, v in target._prefix_payload(logits, features).items()}

    def assert_payload(left, right):
        assert left.keys() == right.keys()
        for name in left:
            torch.testing.assert_close(left[name], right[name], rtol=0, atol=0, msg=name)

    try:
        with torch.inference_mode():
            for tier in ("gpu", "cpu"):
                cache = HybridPrefixCache(256 << 20 if tier == "gpu" else 0, 512 << 20)
                target.cache = cache
                prefix, suffix = ids[: len(ids) // 2], ids[len(ids) // 2 :]
                target.prefill(prefix)
                prefix_features = torch.cat(target.pending_features)
                logits, features = target._forward(suffix, prefill=True)
                split = snapshot(logits, torch.cat([prefix_features, features]))
                # Destroy live KV and all recurrent states before restoring.
                target.prefill([123, 456, 789, 234])
                target.prefill(ids)
                cached = cache.entries[tuple(ids)].tensors
                restored = snapshot(
                    cached["last_logits"].to(target.device), torch.cat(target.pending_features)
                )
                assert_payload(split, restored)
                report["checks"][f"{tier}_restore_exact"] = True

            target.cache = None
            target.prefill([111, 222])  # ensure the next call clears prior state
            target.history = []
            target.gdn.on_table_slot_allocated(0)
            target.gdn.prepare_state_slots()
            full_logits, _ = target._forward(ids, prefill=True)
            full_logits = full_logits.clone()
            report["full_vs_split"] = {
                "max_abs_logits": (full_logits - split["last_logits"]).abs().max().item(),
                "relative_l2_logits": (
                    (full_logits - split["last_logits"]).norm() / full_logits.norm()
                ).item(),
                "same_next_token": full_logits.argmax(-1).item()
                == split["last_logits"].argmax(-1).item(),
            }

            target.prefill(ids[:64])
            checkpoint = target.checkpoint()
            block = ids[64:80]
            target.verify(block)
            target.restore(checkpoint)
            logits, features = target._forward(block[:3])
            replay = snapshot(logits, features)
            target.prefill(ids[:64])
            logits, features = target._forward(block[:3])
            control = snapshot(logits, features)
            assert_payload(replay, control)
            report["checks"]["rollback_and_replay_exact"] = True
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
