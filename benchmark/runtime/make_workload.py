"""Deterministic synthetic token workload with shared prefixes and changing contexts."""

import argparse
import json
import random
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", required=True)
p.add_argument("--lengths", default="128,512,2048")
p.add_argument("--requests-per-length", type=int, default=12)
p.add_argument("--output-tokens", type=int, default=128)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
rng = random.Random(args.seed)
rows = []
for length in map(int, args.lengths.split(",")):
    if length < 2:
        raise ValueError("Context lengths must be at least two")
    shared = [rng.randrange(100, 10000) for _ in range(length // 2)]
    for i in range(args.requests_per_length):
        prefix = shared if i % 3 else [rng.randrange(100, 10000) for _ in shared]
        # Store some exact shared-prefix requests so longest-prefix reuse is exercised.
        ids = (
            prefix
            if i % 4 == 0
            else prefix + [rng.randrange(100, 10000) for _ in range(length - len(prefix))]
        )
        rows.append(
            dict(
                input_ids=ids,
                max_new_tokens=args.output_tokens,
                workload_kind="synthetic_shared_prefix",
            )
        )
path = Path(args.output)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
