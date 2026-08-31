"""Real-weight, same-history decode/verify/rollback numerical regression.

Checks exact logits, draft features, valid KV and GDN state. This is separate
from generation benchmarks: a successful process alone is not a parity test.
"""

import argparse
import json
from pathlib import Path

import torch
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine, EngineConfig
from minisgl.runtime.workload import prepare_workload
from minisgl.speculative.batch import BatchedTargetExecutor
from minisgl.speculative.target import MiniSGLTarget
from minisgl.utils import load_tokenizer


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True, help="Read feature tap configuration only")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads((Path(args.draft) / "config.json").read_text())
    taps = tuple(config["dflash_config"]["target_layer_ids"])
    engine = Engine(
        EngineConfig(
            model_path=args.model,
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            max_running_req=4,
            attention_backend="fi",
            cuda_graph_max_bs=0,
            page_size=256,
            max_seq_len_override=1024,
            num_page_override=16,
            prefix_state_budget_bytes=0,
            gdn_extend_backend="packed",
            memory_budget_bytes=24 << 30,
            runtime_workspace_bytes=1 << 30,
        )
    )
    try:
        executor = BatchedTargetExecutor(engine, taps, 4, cuda_graph=True, target_numerics="stable")
        # Keep diagnostic clones/comparisons on the same stream as model writes.
        torch.cuda.set_stream(engine.stream)
        targets = [
            MiniSGLTarget(engine, capture_layer_ids=taps, slot=i, executor=executor)
            for i in range(4)
        ]
        rows, _ = prepare_workload(
            (Path(__file__).parent / "workloads/chat-long4.jsonl").read_bytes(),
            load_tokenizer(args.model),
            repeat=1,
            chat_template=True,
        )
        anchors = executor.prefill(targets, [r["input_ids"] for r in rows])
        # Decode real histories before comparing alternative execution shapes.
        for _ in range(37):
            out = executor.forward([(t, [a]) for t, a in zip(targets, anchors)])
            anchors = [int(x[0].argmax()) for x in out]
        before = executor.checkpoint(targets)
        tokens = [[anchor] + row["input_ids"][-15:] for anchor, row in zip(anchors, rows)]

        def snapshot():
            states = executor.checkpoint(targets)
            kv = {
                (i, lid, kind): t._kv_view(lid, kind)[: t.length].clone()
                for i, t in enumerate(targets)
                for lid in engine.kv_cache._layer_mapping
                for kind in ("k", "v")
            }
            return states, kv

        def same_state(expected):
            actual, kv = snapshot()
            for a, b in zip(actual, expected[0]):
                assert a.history == b.history
                for lid in a.states:
                    for x, y in zip(a.states[lid], b.states[lid]):
                        torch.testing.assert_close(x, y, rtol=0, atol=0)
            for key, value in kv.items():
                torch.testing.assert_close(value, expected[1][key], rtol=0, atol=0)

        cases = []
        for slots, lengths in [
            ([0, 1, 2, 3], [8, 8, 8, 8]),
            ([3, 0, 2, 1], [16, 2, 8, 4]),
            ([2], [8]),
            ([1, 3, 0], [1, 4, 2]),
        ]:
            items = [(targets[s], tokens[s][:n]) for s, n in zip(slots, lengths)]
            executor.restore(list(zip(targets, before)))
            executor.graph_enabled = True
            logits, features = [[] for _ in items], [[] for _ in items]
            for pos in range(max(lengths)):
                active = [i for i, n in enumerate(lengths) if n > pos]
                out = executor.forward([(items[i][0], [items[i][1][pos]]) for i in active])
                for i, (logit, feature) in zip(active, out):
                    logits[i].append(logit)
                    features[i].append(feature)
            expected = [(torch.cat(a), torch.cat(b)) for a, b in zip(logits, features)]
            expected_state = snapshot()
            executor.restore(list(zip(targets, before)))
            executor.graph_enabled = False
            actual = executor.forward(items)
            for a, b in zip(actual, expected):
                for x, y in zip(a, b):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
            same_state(expected_state)
            # Reject down to different accepted lengths, replay only those slots,
            # and compare against independently restored one-token execution.
            rejected = [(target, block[: max(1, len(block) // 2)]) for target, block in items]
            executor.restore(list(zip(targets, before)))
            executor.forward(rejected)
            replay_state = snapshot()
            executor.restore(list(zip(targets, before)))
            executor.verify(rejected, sequential=True)
            same_state(replay_state)
            cases.append(
                dict(
                    slots=slots,
                    lengths=lengths,
                    exact_logits_features_kv_gdn=True,
                    rollback_replay_exact=True,
                )
            )
        Path(args.output).write_text(
            json.dumps(
                dict(
                    measured=True,
                    target_numerics="stable",
                    gpu=torch.cuda.get_device_name(),
                    cases=cases,
                    execution=executor.stats(),
                ),
                indent=2,
            )
            + "\n"
        )
        print("PASS: real-weight decode/verify/graph/rollback exact parity", flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
