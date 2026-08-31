"""Compare eager/graph logits, features and state at identical real-model histories."""

import argparse
import json
from pathlib import Path

import torch
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine, EngineConfig
from minisgl.speculative.batch import BatchedTargetExecutor
from minisgl.speculative.target import MiniSGLTarget
from minisgl.utils import load_tokenizer


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True, help="Read tap configuration only")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    draft = json.loads((Path(args.draft) / "config.json").read_text())
    taps = tuple(draft["dflash_config"]["target_layer_ids"])
    config = EngineConfig(
        model_path=args.model,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        max_running_req=4,
        attention_backend="fi",
        cuda_graph_max_bs=0,
        page_size=256,
        max_seq_len_override=256,
        num_page_override=4,
        prefix_state_budget_bytes=0,
        gdn_extend_backend="packed",
        memory_budget_bytes=24 << 30,
        runtime_workspace_bytes=1 << 30,
    )
    engine = Engine(config)
    try:
        executor = BatchedTargetExecutor(engine, taps, 4, cuda_graph=True)
        targets = [
            MiniSGLTarget(engine, capture_layer_ids=taps, slot=i, executor=executor)
            for i in range(4)
        ]
        tokenizer = load_tokenizer(args.model)
        for i, target in enumerate(targets):
            target.prefill(tokenizer.encode("Explain KV caching. " * (i + 1)))
        cases = []
        for slots in [(0, 1, 2, 3), (3, 0, 2), (2,), (1, 3, 0, 2)]:
            before = [t.checkpoint() for t in targets]
            items = [(targets[i], [100 + i]) for i in slots]
            executor.graph_enabled = False
            eager = executor.forward(items)
            after_eager = [t.checkpoint() for t in targets]
            kv_eager = {
                (i, lid, kind): targets[i]._kv_view(lid, kind)[targets[i].length - 1].clone()
                for i in slots
                for lid in engine.kv_cache._layer_mapping
                for kind in ("k", "v")
            }
            for t, saved in zip(targets, before):
                t.restore(saved)
            executor.graph_enabled = True
            graphed = executor.forward(items)
            for (a, fa), (b, fb) in zip(eager, graphed):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(fa, fb, rtol=0, atol=0)
            for i, target in enumerate(targets):
                actual = target.checkpoint()
                expected = after_eager[i] if i in slots else before[i]
                assert actual.history == expected.history
                for lid in actual.states:
                    for a, b in zip(actual.states[lid], expected.states[lid]):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
            for (i, lid, kind), expected in kv_eager.items():
                torch.testing.assert_close(
                    targets[i]._kv_view(lid, kind)[targets[i].length - 1],
                    expected,
                    rtol=0,
                    atol=0,
                )
            # Returned features must not alias the mutable graph output buffer.
            saved = [(a.clone(), b.clone()) for a, b in graphed]
            executor.forward(items)
            for (a, fa), (b, fb) in zip(graphed, saved):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(fa, fb, rtol=0, atol=0)
            cases.append(
                dict(
                    slots=slots,
                    exact_logits_features_kv_gdn=True,
                    inactive_slots_unchanged=True,
                    output_buffers_independent=True,
                )
            )
        output = dict(
            measured=True, gpu=torch.cuda.get_device_name(), cases=cases, execution=executor.stats()
        )
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(json.dumps(output))
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
