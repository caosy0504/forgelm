# ForgeLM

ForgeLM is a single-GPU, decoder-only language model training project built around a controlled data-sampling experiment. It covers dataset preparation, a train-only byte-level BPE tokenizer, model training, checkpoint recovery, document-aware evaluation, and greedy decoding with a static KV cache.

The completed WikiText-103 run trained an **88.1M-parameter model on an RTX 4090**. A shared 4,800-step prefix was forked into two equal-budget 1,200-step continuations to compare document-uniform sampling with a transparent quality-tier heuristic. Model selection used full-corpus validation; the selected checkpoint was evaluated once on the held-out test split.

## System

| Component | Implementation |
|---|---|
| Data | Pinned WikiText-103 revision; article boundaries; Unicode normalization; limited PII masking; rule filtering; exact and MinHash/LSH near deduplication; train-to-evaluation decontamination |
| Tokenizer | Deterministic byte-level BPE trained only on prepared training documents; 4,096-token vocabulary in the measured run |
| Model | Pre-norm Transformer with RoPE, RMSNorm, QK-Norm, SwiGLU, tied embeddings, scaled residual initialization, causal SDPA, and a small Z-loss term |
| Training | BF16, gradient accumulation and clipping, fused AdamW, warmup–stable–decay schedule, resumable checkpoints, and SHA-256 run fingerprints |
| Experiment | Shared parent checkpoint; equal continuation token budgets; only the document-sampling policy changes between branches |
| Evaluation | Document-bounded rolling full-corpus likelihood, cross-entropy, perplexity, and bits per byte; validation-only checkpoint selection |
| Decoding | Preallocated per-layer static KV cache; cached/full-forward greedy-output parity and latency/throughput measurements |

The quality score measures prose *form*: useful length, alphabetic content, sentence length, and lexical/trigram diversity in fixed-size word windows. It does **not** measure factual accuracy, safety, or semantic value. Eligible training windows are divided into three tiers of approximately equal window mass; the quality branch samples those tiers with probabilities 0.1/0.3/0.6.

## Completed experiment

| Setting | Value |
|---|---:|
| Dataset | WikiText-103 raw, pinned revision; first 400M raw training characters |
| Prepared training documents / tokens | 21,531 / 180,839,589 |
| Model | 12 layers, width 768, 12 heads, FFN width 2,048; 88,101,120 parameters |
| Context / effective batch | 1,024 tokens / 32,768 sampled tokens per optimizer step |
| Schedule | 4,800 shared steps + 1,200 steps per branch; seed 42 |
| Hardware / precision | RTX 4090 / BF16 |

Both branches started from the **same step-4,800 checkpoint**, tokenizer, dataset, optimizer state, random-seed lineage, and 6,000-step learning-rate schedule. Each continuation sampled 39,321,600 tokens.

| Full-corpus validation | Uniform | Quality-tier |
|---|---:|---:|
| Cross-entropy loss | **1.607508** | 1.617352 |
| Perplexity | **4.990360** | 5.039725 |

The quality-tier branch had **0.99% higher** validation perplexity in this run. The validation-selected uniform checkpoint achieved **5.035705 test perplexity** and **1.078290 test bits per byte** in one held-out full-corpus evaluation. The test was not used for model selection.

Static KV caching generated exactly the same greedy output as full-forward decoding. On the selected RTX 4090 checkpoint, a 7-token prompt followed by 128 new tokens gave median rates of **180.97 versus 173.85 generated tokens/s** across five trials, a **1.04×** ratio. This short-output result does not establish a general decoding speedup.

These are results from **one training seed**. The small sampling difference does not establish a population-level advantage for either policy; human labels for the quality-tier audit have not been completed.

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

## Verification and scope

- Tests cover tokenizer round trips, document-bounded sampling and evaluation, fork controls, attention parity, and cached decoding correctness. The RTX 4090 environment passed 38 tests before training.
- Training-time validation samples fixed random windows; the reported comparison and final test score **every document token once** with rolling context and EOS conditioning at each document boundary. These two evaluation modes should not be compared as if they were identical.
- The measured profile used BF16, SDPA, and fused AdamW. Compilation and activation checkpointing are implemented options but were disabled in this run. An eligible Flash Attention probe does not prove every training forward used that kernel.
- Model-quality filtering, external-corpus evaluation, and scaling experiments are available as separate workflows; they were not part of this measured sampling comparison.
- ForgeLM is a base language model, not an instruction-tuned assistant. Dataset and checkpoint files are excluded from Git. The upstream dataset's terms remain with its authors.

ForgeLM is independent educational engineering work informed by Stanford CS336, Stanford CS224N, and public language-model training literature. Source code is MIT licensed.
