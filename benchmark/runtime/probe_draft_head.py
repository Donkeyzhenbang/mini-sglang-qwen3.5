"""Isolate strided versus flattened DFlash vocabulary projection under graphs.

Synthetic activations/weights use checkpoint dimensions; these are kernel
measurements, not model quality or end-to-end throughput measurements.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.nn import functional as F


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads((Path(args.model) / "config.json").read_text())
    config = config.get("text_config", config)
    torch.manual_seed(917)
    device, dtype = "cuda", torch.bfloat16
    h, vocab = config["hidden_size"], config["vocab_size"]
    weight = torch.randn(vocab, h, device=device, dtype=dtype) / h**0.5
    results = []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for batch in (1, 2, 3, 4):
            x = torch.randn(batch, 8, h, device=device, dtype=dtype)
            timings, outputs = {}, {}
            for name in ("strided", "flat"):
                def project():
                    selected = x[:, 1:]
                    if name == "flat":
                        selected = selected.reshape(-1, h)
                    return F.linear(selected, weight).view(batch, 7, vocab)

                for _ in range(3):
                    project()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    logits = project()
                samples = []
                for _ in range(20):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record(stream)
                    graph.replay()
                    end.record(stream)
                    end.synchronize()
                    samples.append(start.elapsed_time(end))
                timings[name] = dict(median_ms=statistics.median(samples), samples_ms=samples)
                outputs[name] = logits.clone()
                del graph
            a, b = outputs["strided"], outputs["flat"]
            results.append(dict(
                batch=batch, block=8, timings=timings,
                top1_equal=int((a.argmax(-1) == b.argmax(-1)).sum()),
                top1_total=batch * 7,
                max_abs_logit_delta=float((a.float() - b.float()).abs().max()),
            ))
    payload = dict(
        gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        hidden_size=h, vocab_size=vocab, synthetic=True, cuda_graph=True, results=results,
    )
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
