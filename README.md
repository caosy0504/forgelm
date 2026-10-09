# ForgeLM

ForgeLM is a reproducible, single-GPU language-model pretraining project. One decoder-only Transformer, one train-only tokenizer, and a versioned dataset feed a controlled experiment: after a shared training prefix, two branches use equal token budgets to compare document-uniform and quality-tier sampling. The selected checkpoint is then evaluated on the held-out test split and benchmarked with and without static KV caching.

**Status:** The integrated workflow, tests, and CPU end-to-end smoke run are complete. The WikiText-103 RTX 3070 Ti experiment is configured but has **not** been run or measured on that GPU. No WikiText perplexity or GPU speedup is claimed yet.

**Unreleased local profile:** `configs/wikitext4090_*.toml` targets a rented RTX 4090 with an approximately 88M-parameter model and a 400M-character WikiText-103 train prefix. The Mac data/token cache is prepared and audited, but the profile has no RTX 4090 measurements. It is not part of the published v0.3.0 GitHub release; local training instructions are kept outside the public source package.

## Research question

Under an 8 GB single-GPU constraint, can a transparent document-sampling policy improve validation language-model quality at the same training token budget, and can static KV caching improve greedy decoding throughput without changing the generated tokens?

```text
Pinned WikiText-103 splits
  → article-level preparation → normalization, quality rules, deduplication, decontamination
  → train-only byte-level BPE → ForgeLM Transformer → shared-prefix checkpoint
                                             ├── document-uniform continuation
                                             └── quality-tier continuation
  → full-corpus validation → one selected checkpoint → sealed full-corpus test
  → same-checkpoint full-forward / static-KV decode benchmark
```

## Implemented system

| Layer | Implementation |
|---|---|
| Data | Pinned WikiText download option, article boundaries, deterministic normalization, PII masking, exact and near deduplication, train-to-evaluation decontamination, hashed manifests, and a blinded quality-tier audit |
| Tokenization | Deterministic byte-level BPE fitted only on prepared training documents; ranked merge encoding and an optional vocabulary-size audit |
| Model | Decoder-only Transformer with RMSNorm, RoPE, SwiGLU, optional QK-Norm and Z-loss, tied embeddings, scaled residual initialization, and eager/SDPA attention |
| Training | AdamW, warmup-cosine or WSD schedule, gradient accumulation, clipping, BF16 option, activation checkpointing, optional compilation, document-bounded fixed evaluation batches, checkpoint recovery, and sampled-token/reuse/throughput accounting |
| Controlled fork | Copies the prepared data, tokenizer, and verified token cache; checks parent checkpoint, model, dataset, tokenizer, source tree, optimizer schedule, and training configuration; allows only the sampling policy to differ |
| Evaluation | Fixed-seed document-bounded random windows during training; document-aware rolling windows for full validation and final test; cross-entropy, PPL, and full-corpus BPB |
| Inference | Preallocated output and per-layer static KV cache, context-window rebuild, cached/full-forward logit parity, exact greedy-output parity, first-token and total-latency benchmarks |

The quality score is a transparent prose-shape heuristic based on minimum useful length, alphabetic content, sentence length, and lexical/trigram diversity measured in fixed-size windows. Fixed windows avoid automatically penalizing long articles for having a lower whole-document type/token ratio. It is **not** a factuality or safety score. Documents are grouped into three tiers by eligible token-window mass. The uniform branch samples every eligible document window equally; the quality branch first samples a tier, then samples a window uniformly within that tier.

## Controlled experiment

The RTX profile uses a roughly 26.75M-parameter model, 2,048-token vocabulary target, 512-token training sequences, and 1,000 configured optimizer steps. Compilation is disabled in this Windows-oriented profile; BF16, activation checkpointing, and fused AdamW remain configurable. These are **planned settings**, subject to a short GPU memory and throughput check before formal training.

1. Run the base configuration to step 800, keeping the 1,000-step learning-rate schedule intact.
2. Fork the exact step-800 checkpoint into document-uniform and quality-tier branches. Both complete the same remaining 200 steps with the same model, optimizer state, seed, learning-rate schedule, dataset, tokenizer, and sampled-token budget.
3. Score both branches over every validation next-token target, resetting context at each document. `compare-sampling` rejects unmatched controls and chooses the lower-loss checkpoint using validation only.
4. After selection is frozen, run the full-corpus test once on the selected checkpoint. Benchmark cached and uncached greedy decoding on that same checkpoint.

One training seed can establish a controlled local comparison but cannot establish statistical significance across independent training runs. If the result is small, repeat the full lineage with additional seeds before making a general improvement claim.

## Install

Python 3.11–3.13 and a compatible PyTorch installation are required. Select the CUDA-enabled PyTorch build appropriate for the target system from the [official PyTorch installer](https://pytorch.org/get-started/locally/), then install the package and optional data dependency:

```bash
python -m pip install -e ".[data]"
python -m unittest discover -s tests -v
```

The repository does not redistribute WikiText. The preparation command pins `Salesforce/wikitext`, `wikitext-103-raw-v1`, revision `5fddba447aa4e75996922ea0d6b18b42f0a81cc4` and writes source/output hashes. The upstream dataset card reports CC-BY-SA-3.0 and GFDL terms.

## Reproduce the WikiText experiment

Run these commands from the repository root. The three configurations differ only in project name, output directory, and sampling policy.

```bash
forgelm prepare-wikitext --download \
  --raw-dir data/wikitext103_raw \
  --output-dir data/wikitext103_articles \
  --max-train-characters 200000000

forgelm prepare-experiment --config configs/wikitext3070ti_base.toml
forgelm audit-quality --config configs/wikitext3070ti_base.toml --sample-per-tier 20
# Optional compression audit; changing vocabulary requires a new model experiment.
forgelm audit-tokenizer --config configs/wikitext3070ti_base.toml \
  --vocab-sizes 1024 2048 4096

forgelm doctor --config configs/wikitext3070ti_base.toml
forgelm benchmark-system --config configs/wikitext3070ti_base.toml \
  --batch-size 1 --seq-len 256 --warmup 2 --iterations 5
forgelm run --config configs/wikitext3070ti_base.toml --stop-step 800

forgelm run --config configs/wikitext3070ti_uniform.toml \
  --fork artifacts/wikitext3070ti/base/checkpoint_last.pt
forgelm run --config configs/wikitext3070ti_quality.toml \
  --fork artifacts/wikitext3070ti/base/checkpoint_last.pt

forgelm evaluate --config configs/wikitext3070ti_uniform.toml --full-corpus
forgelm evaluate --config configs/wikitext3070ti_quality.toml --full-corpus
forgelm compare-sampling \
  --uniform-dir artifacts/wikitext3070ti/uniform \
  --quality-dir artifacts/wikitext3070ti/quality \
  --output artifacts/wikitext3070ti/comparison_summary.json
```

Inspect `comparison_summary.json` and use the selected branch's configuration for the final test. `forgelm test` always uses full-corpus coverage. For example, **only if quality was selected**:

```bash
forgelm test --config configs/wikitext3070ti_quality.toml
forgelm benchmark-decode \
  --checkpoint artifacts/wikitext3070ti/quality/checkpoint_last.pt \
  --tokenizer artifacts/wikitext3070ti/quality/tokenizer.json \
  --prompt "The history of science" --new-tokens 128 --trials 5 --device cuda \
  --output artifacts/wikitext3070ti/quality/decode_benchmark.json
```

Keep the source tree at the same commit across the shared prefix and forks. Prepared input/output hashes and the sampling-code hash guard cache reuse. A strict `--resume` requires an identical run fingerprint; `--fork` creates a new lineage and permits only the sampling-policy change. Run outputs and corpora are excluded from Git.

`audit-quality` writes a local CSV with document previews and blank `human_label` cells. Label a small sample blindly as 0/1/2, then rerun it with `--labels artifacts/wikitext3070ti/base/quality_audit_samples.csv` to inspect human-label means by hidden tier. The audit is diagnostic: if tiers do not align with judged prose utility, revise the heuristic and restart in a new experiment directory before formal training.

For a separately licensed, versioned external JSONL corpus (one `{"doc_id":"...","text":"...","source":"...","pii_replacements":{}}` document per line), `forgelm evaluate-external --checkpoint ... --tokenizer ... --jsonl ... --seq-len 512 --device cuda` reports document-aware BPB without touching the model-selection validation or final test. After repeating the complete experiment with independent seeds, `forgelm summarize-seeds --comparisons <run1.json> <run2.json> --output <summary.json>` gives descriptive paired results.

## Local end-to-end smoke check

The checked-in sample corpus supports a short CPU verification of the same workflow:

```bash
forgelm prepare-experiment --config configs/integrated_smoke_base.toml
forgelm audit-quality --config configs/integrated_smoke_base.toml --sample-per-tier 2
forgelm run --config configs/integrated_smoke_base.toml --stop-step 2
forgelm run --config configs/integrated_smoke_uniform.toml \
  --fork artifacts/integrated-smoke/base/checkpoint_last.pt
forgelm run --config configs/integrated_smoke_quality.toml \
  --fork artifacts/integrated-smoke/base/checkpoint_last.pt
forgelm evaluate --config configs/integrated_smoke_uniform.toml --full-corpus
forgelm evaluate --config configs/integrated_smoke_quality.toml --full-corpus
forgelm compare-sampling \
  --uniform-dir artifacts/integrated-smoke/uniform \
  --quality-dir artifacts/integrated-smoke/quality \
  --output artifacts/integrated-smoke/comparison_summary.json
```

Smoke results verify the pipeline and controls; their tiny corpus and model are not evidence of language-model quality or GPU performance.

## Verification and limitations

The test suite covers tokenizer parity, document-bounded sampling and evaluation, fork constraints, per-document full-corpus target coverage, quality/tokenizer audits, attention parity, static-cache storage reuse, cached/full-forward logits, and cached/uncached greedy generation across a context-window rebuild. CI also runs the original package checks and the integrated CPU workflow.

- The RTX profile and its memory footprint require measurement on the target machine. If it does not fit, reduce micro-batch size and increase accumulation before changing the model or token budget; record the final configuration before the formal comparison.
- The forced Flash Attention probe checks kernel eligibility for the benchmark shape; it does not prove that every training forward uses that kernel.
- Full-corpus evaluation scores each document token once. The first token is conditioned on EOS; rolling windows overlap by default with a stride of half the evaluation sequence length. No validation/test context crosses documents. BPB includes EOS prediction cost and divides by prepared UTF-8 document bytes.
- The quality heuristic may fail or worsen results. The comparison report preserves negative outcomes.
- The model is a base language model. Interactive completion is not instruction tuning or a reliable assistant.
- No claim is made about production-scale deduplication, factuality, safety, or cross-seed significance.

## Provenance

ForgeLM is independent educational engineering work informed by Stanford CS336, Stanford CS224N, and the author's earlier `mingpt-2` experiments. Concepts such as document-quality sampling, full-corpus evaluation, and cache correctness checks were reworked into a single ForgeLM model and experiment lineage. The data and evaluation design also draws on [DCLM](https://arxiv.org/abs/2406.11794), [SmolLM3's training report](https://huggingface.co/blog/smollm3), and [lm-evaluation-harness's document-aware rolling likelihood](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/model.py). Source code is MIT licensed; dataset licenses remain with their owners.
