# TinyStories 5M Split

This directory is a deterministic 90/5/5 document-level split of the 5M-character TinyStories sample bundled with the Stanford CS336 assignment fixtures.

- Source: https://huggingface.co/datasets/roneneldan/TinyStories
- Paper: https://arxiv.org/abs/2305.07759
- License: CDLA-Sharing-1.0
- Split unit: complete story delimited by `<|endoftext|>`
- Exact duplicates are removed globally before splitting.
- The first and trailing partial fragments in the course sample are discarded.

See `manifest.json` for counts and SHA-256 fingerprints.
