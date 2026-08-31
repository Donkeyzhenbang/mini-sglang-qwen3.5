"""Resolve text/chat inputs without hiding the exact tokens used by an experiment."""

from __future__ import annotations

import hashlib
import json


def prepare_workload(raw, tokenizer, *, repeat=1, chat_template=False):
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not rows or repeat < 1:
        raise ValueError("Workload and repeat count must be nonempty/positive")
    prepared = []
    for row in rows:
        messages = row.get("messages")
        if "input_ids" in row:
            ids = list(row["input_ids"])
        elif messages is not None or chat_template:
            messages = messages or [{"role": "user", "content": row["prompt"]}]
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        count = int(row.get("max_new_tokens", 64))
        if not ids or count < 1:
            raise ValueError("Workload requires input tokens and max_new_tokens >= 1")
        prompt = row.get("prompt")
        if prompt is None:
            prompt = (
                "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                if messages
                else tokenizer.decode(ids, skip_special_tokens=False)
            )
        prepared.append(dict(input_ids=ids, max_new_tokens=count, prompt=prompt))
    prepared *= repeat
    # Preserve existing raw-file hashes for previously supported workloads.
    transformed = repeat != 1 or chat_template or any("messages" in r for r in rows)
    effective = (
        json.dumps(
            [(r["input_ids"], r["max_new_tokens"]) for r in prepared], separators=(",", ":")
        ).encode()
        if transformed
        else raw
    )
    return prepared, hashlib.sha256(effective).hexdigest()


def describe_result(result, row, tokenizer, cache_event):
    eos = getattr(tokenizer, "eos_token_id", None)
    finish_reason = "eos" if result["token_ids"][-1] == eos else "length"
    result.update(
        prompt=row["prompt"],
        prompt_token_ids=row["input_ids"],
        output_text=tokenizer.decode(result["token_ids"], skip_special_tokens=True),
        cache_event=dict(cache_event),
        finish_reason=finish_reason,
    )
    return result


def print_result(index, result):
    event = result["cache_event"]
    print(
        f"\nRequest {index + 1} | cache={event['status']} "
        f"matched={event['matched_tokens']}/{event['prompt_tokens']} "
        f"tier={event['tier']} stored={event['stored']} finish={result['finish_reason']}",
        flush=True,
    )
    print(f"Prompt: {result['prompt']}\nAnswer: {result['output_text']}", flush=True)
