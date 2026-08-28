from __future__ import annotations

import argparse
import json

from forgelm.dataset_split import split_tinystories_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic TinyStories train/validation/test splits")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", default="data/tinystories_5m")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(split_tinystories_file(args.source, args.output_dir, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()

