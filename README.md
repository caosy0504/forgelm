# ForgeLM

ForgeLM is a single-GPU, decoder-only language model training project built around a controlled data-sampling experiment. It covers dataset preparation, a train-only byte-level BPE tokenizer, model training, checkpoint recovery, document-aware evaluation, and greedy decoding with a static KV cache.

Two completed WikiText-103 experiments trained an **88.1M-parameter model on an RTX 4090**. Each used a 4,800-step shared prefix and equal-budget 1,200-step continuations. The first compared document-uniform sampling with a quality-tier heuristic; the second also tested a length-balanced tier assignment. Each experiment selected its checkpoint on full-corpus validation before one held-out test evaluation.

## System

| Component | Implementation |
|---|---|
| Data | Pinned WikiText-103 revision; article boundaries; Unicode normalization; limited PII masking; rule filtering; exact and MinHash/LSH near deduplication; train-to-evaluation decontamination |
| Tokenizer | Deterministic byte-level BPE trained only on prepared training documents; 4,096-token vocabulary in the measured run |
| Model | Pre-norm Transformer with RoPE, RMSNorm, QK-Norm, SwiGLU, tied embeddings, scaled residual initialization, causal SDPA, and a small Z-loss term |
| Training | BF16, gradient accumulation and clipping, fused AdamW, warmup–stable–decay schedule, resumable checkpoints, and SHA-256 run fingerprints |
| Experiment | Shared parent checkpoint; equal continuation token budgets; document-uniform, quality-tier, and length-balanced quality-tier sampling policies |
| Evaluation | Document-bounded rolling full-corpus likelihood, cross-entropy, perplexity, and bits per byte; validation-only checkpoint selection |
| Decoding | Preallocated per-layer static KV cache; cached/full-forward greedy-output parity and latency/throughput measurements |

The quality score measures prose *form*: useful length, alphabetic content, sentence length, and lexical/trigram diversity in fixed-size word windows. It does **not** measure factual accuracy, safety, or semantic value. Eligible training windows are divided into three tiers of approximately equal window mass; the quality branch samples those tiers with probabilities 0.1/0.3/0.6.

## Completed experiments

| Setting | Value |
|---|---:|
| Dataset | WikiText-103 raw, pinned revision; first 400M raw training characters |
| Prepared training documents / tokens | 21,531 / 180,839,589 |
| Model | 12 layers, width 768, 12 heads, FFN width 2,048; 88,101,120 parameters |
| Context / effective batch | 1,024 tokens / 32,768 sampled tokens per optimizer step |
| Schedule | 4,800 shared steps + 1,200 steps per branch; independent seeds 42 and 43 |
| Hardware / precision | RTX 4090 / BF16; seed 43 used a 48 GB variant without changing the training configuration |

Within each experiment, branches started from the **same step-4,800 checkpoint**, tokenizer, dataset, optimizer state, random-seed lineage, and 6,000-step learning-rate schedule. Each continuation sampled 39,321,600 tokens.

### Seed 42: initial sampling comparison

| Full-corpus validation | Uniform | Quality-tier |
|---|---:|---:|
| Cross-entropy loss | **1.607508** | 1.617352 |
| Perplexity | **4.990360** | 5.039725 |

The quality-tier branch had **0.99% higher** validation perplexity in this run. The validation-selected uniform checkpoint achieved **5.035705 test perplexity** and **1.078290 test bits per byte** in one held-out full-corpus evaluation. The test was not used for model selection.

Static KV caching generated exactly the same greedy output as full-forward decoding. On the selected RTX 4090 checkpoint, a 7-token prompt followed by 128 new tokens gave median rates of **180.97 versus 173.85 generated tokens/s** across five trials, a **1.04×** ratio. This short-output result does not establish a general decoding speedup.

### Seed 43: length-balanced follow-up

The follow-up first grouped eligible documents into token-length quartiles, then assigned quality tiers by eligible training-window mass within each quartile. This controlled an observed length bias in the heuristic tier assignment; it did not introduce human-verified quality labels. The three continuations shared the new seed-43 parent checkpoint and were compared on the same 515,512 validation tokens.

| Full-corpus validation | Uniform | Original quality-tier | Length-balanced quality-tier |
|---|---:|---:|---:|
| Cross-entropy loss | **1.625259** | 1.634462 | 1.632634 |
| Perplexity | **5.079734** | 5.126699 | 5.117334 |

Length balancing reduced the quality branch's validation perplexity by about **0.18%** relative to the original tier assignment, but remained **0.74% higher** than uniform sampling. Validation therefore selected uniform. Its single held-out evaluation scored 583,812 test tokens with **5.112307 perplexity** and **1.088361 bits per byte**. This is a separate seed-43 run; its test score should not be substituted for the seed-42 model's result above.

Both seeds selected uniform over the original quality-tier policy. Two seeds and small differences still do not establish a general advantage. A 60-document, excerpt-only AI-assisted audit was recorded for the length-balanced hypothesis; the human-label field remains empty, and those labels are not human ground truth.

### Post-training two-seed ensemble

The two validation-selected uniform checkpoints share the same tokenizer, model configuration, and byte-identical prepared train/validation/test corpora. Without additional training, `scripts/evaluate_ensemble.py` mixes their **predicted token probabilities** with fixed 0.5/0.5 weights. It scores each document independently with the same 1,024-token rolling context and 512-token stride used above. The script checks checkpoint steps, seeds, tokenizer and corpus hashes, and training settings before evaluation.

| Full-corpus validation, 515,512 tokens | Seed 42 uniform | Seed 43 uniform | Equal-probability ensemble |
|---|---:|---:|---:|
| Perplexity, Mac MPS FP32 | 4.990135 | 5.079340 | **4.808751** |

The ensemble lowered validation perplexity by **3.63%** relative to the better single model, and improved negative log-likelihood on all 60 validation documents. A retrospective test evaluation, with the 0.5/0.5 rule unchanged, gave **4.842947** versus **5.035461** for the better single model on the same Mac FP32 path; all 61 test documents improved. Component-model test results had already been seen during earlier experiments, so this test is reported as retrospective confirmation, not a fresh untouched holdout.

This is an **inference-time ensemble**, not an improved single-model checkpoint. It requires two model forwards and both weight sets at inference; the gain must be weighed against latency and memory cost. The previous single-model training and sampling conclusions remain unchanged.

The reproducible, path-free metric and hash summary is in [`reports/ensemble_seed42_seed43/summary.json`](reports/ensemble_seed42_seed43/summary.json); full document-level reports remain local because they contain machine-specific paths.

## Reproduce

Use Python 3.11–3.13 and a CUDA-enabled PyTorch build compatible with your GPU. The measured environment used Python 3.11.8 and PyTorch 2.11.0+cu128. Install the project after PyTorch:

~~~bash
python -m pip install -e ".[data]"
python -m unittest discover -s tests -q
~~~

From the repository root, prepare the pinned corpus and inspect the data before training:

~~~bash
forgelm prepare-wikitext --download \
  --raw-dir data/wikitext103_400m_raw \
  --output-dir data/wikitext103_400m_articles \
  --max-train-characters 400000000
forgelm prepare-experiment --config configs/wikitext4090_base.toml
forgelm audit-quality --config configs/wikitext4090_base.toml --sample-per-tier 20
forgelm doctor --config configs/wikitext4090_base.toml
~~~

Run the shared prefix and both continuations:

~~~bash
forgelm run --config configs/wikitext4090_base.toml --stop-step 4800
forgelm run --config configs/wikitext4090_uniform.toml \
  --fork artifacts/wikitext4090/base/checkpoint_last.pt
forgelm run --config configs/wikitext4090_quality.toml \
  --fork artifacts/wikitext4090/base/checkpoint_last.pt
forgelm evaluate --config configs/wikitext4090_uniform.toml --full-corpus
forgelm evaluate --config configs/wikitext4090_quality.toml --full-corpus
forgelm compare-sampling \
  --uniform-dir artifacts/wikitext4090/uniform \
  --quality-dir artifacts/wikitext4090/quality \
  --output artifacts/wikitext4090/comparison_summary.json
~~~

Select the branch named by comparison_summary.json **before** using the test set. The measured seed-42 run selected uniform; a new run must follow its own validation comparison. For this run, the final commands were:

~~~bash
forgelm test --config configs/wikitext4090_uniform.toml
forgelm benchmark-decode \
  --checkpoint artifacts/wikitext4090/uniform/checkpoint_last.pt \
  --tokenizer artifacts/wikitext4090/uniform/tokenizer.json \
  --prompt "The history of science" --new-tokens 128 --trials 5 --device cuda \
  --output artifacts/wikitext4090/uniform/decode_benchmark.json
~~~

The measured run used a 100-step preflight and resumed the identical base configuration to step 4,800. Preflight steps were part of the 4,800-step prefix. Keep source code and the three experiment configurations unchanged after training starts: strict resume and controlled forking validate their fingerprints.

To reproduce the seed-43 follow-up, use the four `configs/wikitext4090_lb_seed43_*.toml` files. Train the base to step 4,800, fork `uniform`, `quality_original`, and `quality_lengthbalanced` from its checkpoint, then run `evaluate --full-corpus` on all three and `compare-sampling` twice with uniform as the reference. Freeze the lowest full-validation-loss branch before calling `test` once. The seed-43 source fingerprint differs from seed 42, so its base must be trained independently rather than resumed from the older checkpoint.

After obtaining both selected uniform checkpoints, reproduce the fixed ensemble on validation:

~~~bash
PYTHONPATH=src python scripts/evaluate_ensemble.py \
  --checkpoint /path/to/seed42_uniform.pt \
  --checkpoint /path/to/seed43_uniform.pt \
  --tokenizer artifacts/wikitext4090_lb_seed43/base/tokenizer.json \
  --jsonl artifacts/wikitext4090_lb_seed43/base/dataset/validation.jsonl \
  --split validation --device auto \
  --output artifacts/ensemble_seed42_seed43/validation.json
~~~

## Verification and scope

- Tests cover tokenizer round trips, document-bounded sampling and evaluation, fork controls, attention parity, and cached decoding correctness. The seed-42 environment passed 38 tests; the seed-43 environment passed 42 tests before training.
- Training-time validation samples fixed random windows; the reported comparison and final test score **every document token once** with rolling context and EOS conditioning at each document boundary. These two evaluation modes should not be compared as if they were identical.
- The measured profile used BF16, SDPA, and fused AdamW. Compilation and activation checkpointing are implemented options but were disabled in this run. An eligible Flash Attention probe does not prove every training forward used that kernel.
- Model-quality filtering, external-corpus evaluation, and scaling experiments are available as separate workflows; they were not part of this measured sampling comparison.
- ForgeLM is a base language model, not an instruction-tuned assistant. Dataset and checkpoint files are excluded from Git. The upstream dataset's terms remain with its authors.

ForgeLM is independent educational engineering work informed by Stanford CS336, Stanford CS224N, and public language-model training literature. Source code is MIT licensed.
