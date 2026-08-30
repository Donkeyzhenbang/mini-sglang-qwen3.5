"""Pin public model revisions and optionally download weights to the data disk.

Config-only mode is the default. Remote Python model code is never downloaded
or executed by this helper. Full weights require the explicit --weights flag.
"""

import argparse
import json
from pathlib import Path


def main():
    from huggingface_hub import HfApi, snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--target-revision", default="main")
    parser.add_argument("--draft-revision", default="1905b5a")
    parser.add_argument("--weights", action="store_true")
    args = parser.parse_args()
    root = Path(args.directory)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "models.lock.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    entries = []
    for repo, revision in [
        ("Qwen/Qwen3.5-4B", args.target_revision),
        ("z-lab/Qwen3.5-4B-DFlash", args.draft_revision),
    ]:
        pinned = next((e for e in (previous or {}).get("models", []) if e["repo"] == repo), None)
        sha = (
            pinned["revision"]
            if pinned
            else HfApi(token=False).model_info(repo, revision=revision).sha
        )
        patterns = ["config.json"]
        if args.weights:
            patterns += [
                "*.safetensors",
                "*.safetensors.index.json",
                "tokenizer*",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json",
                "added_tokens.json",
                "chat_template*",
                "generation_config.json",
            ]
        folder = root / repo.split("/")[-1]
        snapshot_download(
            repo, revision=sha, allow_patterns=patterns, local_dir=folder, token=False
        )
        entries.append(
            dict(
                repo=repo,
                revision=sha,
                path=str(folder.resolve()),
                weights_downloaded=args.weights
                or bool(pinned and pinned.get("weights_downloaded")),
            )
        )
    manifest_path.write_text(json.dumps(dict(models=entries), indent=2))
    print(manifest_path)


if __name__ == "__main__":
    main()
