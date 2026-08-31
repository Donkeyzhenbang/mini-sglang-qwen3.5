"""Isolate verify GEMM-shape effects from a fixed real-model recurrent state."""

import argparse
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--workload", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    from minisgl.distributed import DistributedInfo
    from minisgl.engine import Engine, EngineConfig
    from minisgl.speculative.target import MiniSGLTarget

    rows = [json.loads(s) for s in Path(args.workload).read_text().splitlines()]
    baseline = json.loads(Path(args.baseline).read_text())["requests"]
    candidate = json.loads(Path(args.candidate).read_text())["requests"]
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
        gdn_extend_backend="packed",
    )
    engine = Engine(config)
    target = MiniSGLTarget(engine)
    records = []
    try:
        with torch.inference_mode():
            for i, (a, b) in enumerate(zip(baseline, candidate)):
                pos = next(
                    (j for j, (u, v) in enumerate(zip(a["token_ids"], b["token_ids"])) if u != v),
                    None,
                )
                if pos is None or pos == 0:
                    continue
                target.prefill(rows[i]["input_ids"])
                for token in a["token_ids"][: pos - 1]:
                    target.verify([token])
                checkpoint = target.checkpoint()
                results = []
                reference_logits = None
                for block in [1, 2, 4, 8, 16]:
                    target.restore(checkpoint)
                    tokens = a["token_ids"][pos - 1 : pos - 1 + block]
                    tokens += [123] * (block - len(tokens))
                    logits, _ = target._forward(tokens)
                    target.synchronize()
                    first = logits[0].clone()
                    if reference_logits is None:
                        reference_logits = first
                    values, ids = first.topk(5)
                    results.append(
                        dict(
                            block=block,
                            argmax=first.argmax().item(),
                            top_ids=ids.tolist(),
                            top_logits=values.tolist(),
                            max_abs_vs_single=(first - reference_logits).abs().max().item(),
                        )
                    )
                records.append(
                    dict(
                        request=i,
                        first_generation_difference=pos,
                        target_token=a["token_ids"][pos],
                        candidate_token=b["token_ids"][pos],
                        fixed_history_shape_probes=results,
                    )
                )
            output = dict(measured=True, probes=records)
            Path(args.output).write_text(json.dumps(output, indent=2))
            print(json.dumps(output))
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
