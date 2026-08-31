"""Save teacher-forced GPU logits/layer taps for an independent HF comparison.

Run HF in its separate Transformers environment, then mini in the runtime env.
Both consume the same token IDs; this is a diagnostic, not a speed benchmark.
"""

import argparse
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["hf", "mini"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True, help="JSON with prompt_ids and continuation_ids")
    p.add_argument("--output", required=True)
    p.add_argument("--gdn-extend", default="recurrent", choices=["recurrent", "packed"])
    args = p.parse_args()
    data = json.loads(Path(args.input).read_text())
    segments = [data["prompt_ids"]] + [[t] for t in data["continuation_ids"]]
    traces = []
    engine = None
    with torch.inference_mode():
        if args.backend == "hf":
            from transformers import Qwen3_5ForConditionalGeneration

            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                args.model, dtype=torch.bfloat16, device_map={"": "cuda"}
            ).eval()
            cache = None
            for ids in segments:
                result = model(
                    torch.tensor([ids], device="cuda"),
                    past_key_values=cache,
                    use_cache=True,
                    output_hidden_states=True,
                )
                cache = result.past_key_values
                traces.append(
                    dict(
                        logits=result.logits[0, -1].cpu(),
                        layers=torch.stack([h[0, -1] for h in result.hidden_states[1:-1]]).cpu(),
                        conv=cache.layers[0].conv_states[0, :, -3:].cpu(),
                        ssm=cache.layers[0].recurrent_states[0].transpose(-1, -2).cpu(),
                    )
                )
        else:
            from minisgl.distributed import DistributedInfo
            from minisgl.engine import Engine, EngineConfig
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
                gdn_extend_backend=args.gdn_extend,
            )
            engine = Engine(config)
            taps = tuple(range(config.model_config.num_layers - 1))
            target = MiniSGLTarget(engine, capture_layer_ids=taps)
            for i, ids in enumerate(segments):
                logits, features = target._forward(ids, prefill=i == 0)
                target.synchronize()
                rt = target.gdn._runtime[0]
                traces.append(
                    dict(
                        logits=logits[-1].cpu(),
                        layers=features[-1].reshape(len(taps), -1).cpu(),
                        conv=rt.conv_cache[0].cpu(),
                        ssm=rt.ssm_cache[0].cpu(),
                    )
                )
    torch.save(dict(backend=args.backend, input=data, traces=traces), args.output)
    if engine:
        engine.shutdown()


if __name__ == "__main__":
    main()
