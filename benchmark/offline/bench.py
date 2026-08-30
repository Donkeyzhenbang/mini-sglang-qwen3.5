# Adapted from: https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py

import time
from random import randint, seed

from minisgl.core import SamplingParams
from minisgl.llm import LLM


def main():
    seed(0)
    num_seqs = 64
    max_input_len = 512
    max_ouput_len = 256

    # align the hyperparameters
    llm = LLM(
        "/root/autodl-tmp/models/models/Qwen--Qwen3.5-0.8B/snapshots/master",
        max_seq_len_override=4096,
        max_extend_tokens=4096,
        cuda_graph_max_bs=0,
        page_size=256,
        memory_ratio=0.5,
    )

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len))
        for _ in range(num_seqs)
    ]
    llm.generate(["Benchmark: "], SamplingParams(temperature=0.1))  # to warm up flashinfer
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
