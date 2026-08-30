"""Compare measured runs, refusing mismatched workloads or token outputs."""
import argparse
import json
from pathlib import Path


def summarize(data):
    requests = data['requests']
    output_tokens = sum(len(r['token_ids']) for r in requests)
    decoded = sum(max(0, len(r['token_ids']) - 1) for r in requests)
    decode_ms = sum(r['decode_ms'] for r in requests)
    total_ms = decode_ms + sum(r['ttft_ms'] for r in requests)
    rounds = [x for r in requests for x in r['rounds']]
    return dict(mode=data['mode'], requests=len(requests), output_tokens=output_tokens,
        output_tokens_per_second=output_tokens * 1000 / max(total_ms, 1e-9),
        decode_tokens_per_second=decoded * 1000 / max(decode_ms, 1e-9),
        mean_ttft_ms=sum(r['ttft_ms'] for r in requests) / max(len(requests), 1),
        aggregate_tpot_ms=decode_ms / max(decoded, 1),
        mean_progress_per_round=sum(x['progress'] for x in rounds) / max(len(rounds), 1),
        draft_ms=sum(x['draft_ms'] for x in rounds),
        verify_ms=sum(x['verify_ms'] for x in rounds),
        restore_ms=sum(x['restore_ms'] for x in rounds),
        peak_allocated_bytes=data['peak_allocated_bytes'], cache=data['cache'])


def compare(runs):
    baseline = runs[0]
    if not all(r.get('measured') is True for r in runs):
        raise ValueError('Only measured GPU runs may be compared')
    for run in runs[1:]:
        for key in ('workload_sha256', 'target_config_sha256', 'gpu'):
            if not baseline.get(key) or baseline[key] != run.get(key):
                raise ValueError(f'Runs differ in {key}')
        if [r['token_ids'] for r in baseline['requests']] != [r['token_ids'] for r in run['requests']]:
            raise ValueError('Greedy token parity failed; performance comparison is invalid')
    return [summarize(run) for run in runs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    args = parser.parse_args()
    print(json.dumps(compare([json.loads(Path(p).read_text()) for p in args.runs]), indent=2))


if __name__ == '__main__':
    main()
