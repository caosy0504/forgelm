from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch

from .benchmark import benchmark_attention_implementations, benchmark_modern_training_stack
from .chat import DEFAULT_SYSTEM_PROMPT, ChatTurn, generate_chat_response
from .comparison import compare_sampling, summarize_seeds
from .config import ProjectConfig
from .data_pipeline import prepare_dataset
from .decode_benchmark import benchmark_decode
from .evaluation import evaluate_checkpoint
from .pipeline import build_quality_model, run_data_ablation, run_pipeline
from .quality_model import train_quality_classifier
from .quality_audit import audit_quality
from .reproducibility import environment_metadata, resolve_device
from .scaling import fit_from_json
from .tokenizer import BytePairTokenizer
from .tokenizer_audit import audit_tokenizers
from .training import load_model_from_checkpoint
from .wikitext import download_wikitext, prepare_wikitext


def print_json(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forgelm", description="ForgeLM training and experimentation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="validate config and report the runtime environment")
    doctor.add_argument("--config", default="configs/smoke.toml")

    prepare = subparsers.add_parser("prepare-data", help="clean, mask, deduplicate, and split a corpus")
    prepare.add_argument("--config", default="configs/smoke.toml")

    prepare_experiment = subparsers.add_parser(
        "prepare-experiment", help="prepare verified data, tokenizer, and token cache without training"
    )
    prepare_experiment.add_argument("--config", default="configs/wikitext3070ti_base.toml")

    wiki = subparsers.add_parser("prepare-wikitext", help="download and prepare article-level WikiText-103 splits")
    wiki.add_argument("--raw-dir", default="data/wikitext103_raw")
    wiki.add_argument("--output-dir", default="data/wikitext103_articles")
    wiki.add_argument("--download", action="store_true")
    wiki.add_argument("--max-train-characters", type=int, default=200_000_000)
    wiki.add_argument("--min-document-chars", type=int, default=200)

    quality = subparsers.add_parser("train-quality", help="train and validate the model-based quality filter")
    quality.add_argument("--config", default="configs/smoke_v2.toml")

    quality_audit = subparsers.add_parser("audit-quality", help="inspect heuristic tiers and prepare blinded labels")
    quality_audit.add_argument("--config", default="configs/wikitext3070ti_base.toml")
    quality_audit.add_argument("--sample-per-tier", type=int, default=20)
    quality_audit.add_argument("--seed", type=int, default=42)
    quality_audit.add_argument("--labels", default=None)

    tokenizer_audit = subparsers.add_parser("audit-tokenizer", help="compare train-only BPE vocabulary sizes")
    tokenizer_audit.add_argument("--config", default="configs/wikitext3070ti_base.toml")
    tokenizer_audit.add_argument("--vocab-sizes", type=int, nargs="+", default=[1024, 2048, 4096])
    tokenizer_audit.add_argument("--max-training-characters", type=int, default=None)

    ablation = subparsers.add_parser("ablate-data", help="run controlled raw/heuristic/dedup/model-quality ablations")
    ablation.add_argument("--config", default="configs/smoke_v2.toml")

    pipeline = subparsers.add_parser("run", help="execute the full data-to-generation pipeline")
    pipeline.add_argument("--config", default="configs/smoke.toml")
    pipeline.add_argument("--resume", default=None, help="trusted ForgeLM checkpoint to resume")
    pipeline.add_argument("--fork", default=None, help="common-prefix checkpoint for a controlled sampling branch")
    pipeline.add_argument("--stop-step", type=int, default=None, help="save a common prefix before configured steps")

    generate = subparsers.add_parser("generate", help="generate text from a trained checkpoint")
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--tokenizer", required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=80)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-k", type=int, default=40)
    generate.add_argument("--device", default="auto")

    evaluate = subparsers.add_parser("evaluate", help="evaluate a trained checkpoint on the prepared validation set")
    evaluate.add_argument("--config", default="configs/rtx3070ti_v2.toml")
    evaluate.add_argument("--checkpoint", default=None)
    evaluate.add_argument("--full-corpus", action="store_true", help="score every next-token target once")
    evaluate.add_argument("--stride", type=int, default=None, help="new scored tokens per rolling window")

    external = subparsers.add_parser("evaluate-external", help="document-aware held-out JSONL evaluation")
    external.add_argument("--checkpoint", required=True)
    external.add_argument("--tokenizer", required=True)
    external.add_argument("--jsonl", required=True)
    external.add_argument("--device", default="auto")
    external.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    external.add_argument("--seq-len", type=int, required=True)
    external.add_argument("--stride", type=int, default=None)
    external.add_argument("--output", default=None)

    final_test = subparsers.add_parser("test", help="run the final held-out test-set evaluation")
    final_test.add_argument("--config", default="configs/rtx3070ti_v2.toml")
    final_test.add_argument("--checkpoint", default=None)
    final_test.add_argument("--stride", type=int, default=None)

    decode_benchmark = subparsers.add_parser("benchmark-decode", help="compare full-forward and static KV decoding")
    decode_benchmark.add_argument("--checkpoint", required=True)
    decode_benchmark.add_argument("--tokenizer", required=True)
    decode_benchmark.add_argument("--prompt", required=True)
    decode_benchmark.add_argument("--new-tokens", type=int, default=128)
    decode_benchmark.add_argument("--trials", type=int, default=5)
    decode_benchmark.add_argument("--device", default="auto")
    decode_benchmark.add_argument("--output", default=None)

    compare = subparsers.add_parser("compare-sampling", help="verify and compare two controlled sampling branches")
    compare.add_argument("--uniform-dir", required=True)
    compare.add_argument("--quality-dir", required=True)
    compare.add_argument("--output", required=True)

    multiseed = subparsers.add_parser("summarize-seeds", help="summarize repeated controlled comparisons")
    multiseed.add_argument("--comparisons", nargs="+", required=True)
    multiseed.add_argument("--output", required=True)

    chat = subparsers.add_parser("chat", help="interactive completion loop for a trained base model")
    chat.add_argument("--config", default="configs/rtx3070ti_v2.toml")
    chat.add_argument("--checkpoint", default=None)
    chat.add_argument("--tokenizer", default=None)
    chat.add_argument("--max-new-tokens", type=int, default=128)
    chat.add_argument("--temperature", type=float, default=0.8)
    chat.add_argument("--top-k", type=int, default=50)
    chat.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)

    benchmark = subparsers.add_parser("benchmark", help="compare eager and PyTorch SDPA attention")
    benchmark.add_argument("--config", default="configs/smoke.toml")
    benchmark.add_argument("--iterations", type=int, default=5)
    benchmark.add_argument("--warmup", type=int, default=2)
    benchmark.add_argument("--batch-size", type=int, default=None)
    benchmark.add_argument("--seq-len", type=int, default=None)

    system_benchmark = subparsers.add_parser(
        "benchmark-system", help="compare the eager FP32 baseline with the configured modern training stack"
    )
    system_benchmark.add_argument("--config", default="configs/smoke_v2.toml")
    system_benchmark.add_argument("--iterations", type=int, default=5)
    system_benchmark.add_argument("--warmup", type=int, default=2)
    system_benchmark.add_argument("--batch-size", type=int, default=None)
    system_benchmark.add_argument("--seq-len", type=int, default=None)

    scaling = subparsers.add_parser("fit-scaling", help="fit IsoFLOPs power laws from run records")
    scaling.add_argument("--runs", default="data/scaling_runs.json")
    scaling.add_argument("--target-compute", type=float, default=2.56e17)
    scaling.add_argument("--output", default="artifacts/scaling_fit.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        config = ProjectConfig.from_toml(args.config)
        device = resolve_device(config.training.device)
        print_json(
            {
                "config": str(Path(args.config).resolve()),
                "input_exists": config.input_path.exists(),
                "artifact_dir": str(config.artifact_dir),
                "environment": environment_metadata(device),
                "status": "ok",
            }
        )
        return 0
    if args.command == "prepare-data":
        config = ProjectConfig.from_toml(args.config)
        classifier, _ = build_quality_model(config, config.artifact_dir)
        print_json(
            prepare_dataset(
                config.input_path,
                config.artifact_dir / "dataset",
                config.data,
                classifier,
                validation_path=config.validation_path,
                test_path=config.test_path,
            ).__dict__
        )
        return 0
    if args.command == "prepare-experiment":
        print_json(run_pipeline(args.config, prepare_only=True))
        return 0
    if args.command == "prepare-wikitext":
        if args.download:
            download_wikitext(args.raw_dir, max_train_characters=args.max_train_characters)
        print_json(prepare_wikitext(args.raw_dir, args.output_dir, min_document_chars=args.min_document_chars))
        return 0
    if args.command == "train-quality":
        config = ProjectConfig.from_toml(args.config)
        seed_path = config.quality_seed_path
        if seed_path is None:
            raise ValueError("the selected config has no quality seed path")
        _, report = train_quality_classifier(
            seed_path,
            feature_dimension=config.data.quality_hash_dim,
            epochs=config.data.quality_epochs,
            learning_rate=config.data.quality_learning_rate,
            threshold=config.data.quality_threshold,
            output_path=config.artifact_dir / "quality_model.json",
        )
        print_json(report.__dict__)
        return 0
    if args.command == "audit-quality":
        print_json(audit_quality(
            args.config, sample_per_tier=args.sample_per_tier, seed=args.seed, labels_path=args.labels
        ))
        return 0
    if args.command == "audit-tokenizer":
        print_json(audit_tokenizers(
            args.config, vocab_sizes=args.vocab_sizes,
            max_training_characters=args.max_training_characters,
        ))
        return 0
    if args.command == "ablate-data":
        print_json(run_data_ablation(args.config))
        return 0
    if args.command == "run":
        print_json(run_pipeline(args.config, resume_path=args.resume, fork_path=args.fork, stop_step=args.stop_step))
        return 0
    if args.command == "benchmark-decode":
        print_json(benchmark_decode(
            checkpoint_path=args.checkpoint,
            tokenizer_path=args.tokenizer,
            prompt=args.prompt,
            new_tokens=args.new_tokens,
            trials=args.trials,
            device_name=args.device,
            output_path=args.output,
        ))
        return 0
    if args.command == "compare-sampling":
        print_json(compare_sampling(args.uniform_dir, args.quality_dir, args.output))
        return 0
    if args.command == "summarize-seeds":
        print_json(summarize_seeds(args.comparisons, args.output))
        return 0
    if args.command == "generate":
        device = resolve_device(args.device)
        tokenizer = BytePairTokenizer.load(args.tokenizer)
        model = load_model_from_checkpoint(args.checkpoint, device)
        prompt_ids = tokenizer.encode(args.prompt)
        if not prompt_ids:
            raise ValueError("prompt produced no tokens")
        generator = torch.Generator(device=device.type).manual_seed(43) if device.type in {"cpu", "cuda"} else None
        generated = model.generate(
            torch.tensor([prompt_ids], dtype=torch.long, device=device),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_id=tokenizer.EOS_ID,
            generator=generator,
        )[0].tolist()
        print(tokenizer.decode(generated))
        return 0
    if args.command == "evaluate":
        config = ProjectConfig.from_toml(args.config)
        checkpoint = Path(args.checkpoint) if args.checkpoint else config.artifact_dir / "checkpoint_last.pt"
        report = evaluate_checkpoint(
            checkpoint_path=checkpoint,
            tokenizer_path=config.artifact_dir / "tokenizer.json",
            validation_jsonl=config.artifact_dir / "dataset" / "validation.jsonl",
            device_name=config.training.device,
            batch_size=config.training.batch_size,
            seq_len=config.training.seq_len,
            eval_batches=config.training.eval_batches,
            seed=config.seed + 10_000,
            precision=config.training.precision,
            split_name="validation",
            output_path=config.artifact_dir / "evaluation_summary.json",
            full_corpus=args.full_corpus,
            stride=args.stride,
        )
        print_json(report)
        return 0
    if args.command == "evaluate-external":
        print_json(evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            tokenizer_path=args.tokenizer,
            validation_jsonl=args.jsonl,
            device_name=args.device,
            batch_size=1,
            seq_len=args.seq_len,
            eval_batches=1,
            seed=0,
            precision=args.precision,
            split_name="external",
            output_path=args.output,
            full_corpus=True,
            stride=args.stride,
        ))
        return 0
    if args.command == "test":
        config = ProjectConfig.from_toml(args.config)
        checkpoint = Path(args.checkpoint) if args.checkpoint else config.artifact_dir / "checkpoint_last.pt"
        test_jsonl = config.artifact_dir / "dataset" / "test.jsonl"
        if not test_jsonl.exists():
            raise FileNotFoundError("prepared test.jsonl not found; configure data.test_path and run training first")
        report = evaluate_checkpoint(
            checkpoint_path=checkpoint,
            tokenizer_path=config.artifact_dir / "tokenizer.json",
            validation_jsonl=test_jsonl,
            device_name=config.training.device,
            batch_size=config.training.batch_size,
            seq_len=config.training.seq_len,
            eval_batches=config.training.eval_batches,
            seed=config.seed + 20_000,
            precision=config.training.precision,
            split_name="test",
            output_path=config.artifact_dir / "test_summary.json",
            full_corpus=True,
            stride=args.stride,
        )
        print("警告：Test set 应在模型、超参数和 checkpoint 全部冻结后才运行。")
        print_json(report)
        return 0
    if args.command == "chat":
        config = ProjectConfig.from_toml(args.config)
        device = resolve_device(config.training.device)
        checkpoint = Path(args.checkpoint) if args.checkpoint else config.artifact_dir / "checkpoint_last.pt"
        tokenizer_path = Path(args.tokenizer) if args.tokenizer else config.artifact_dir / "tokenizer.json"
        tokenizer = BytePairTokenizer.load(tokenizer_path)
        model = load_model_from_checkpoint(checkpoint, device)
        generator = (
            torch.Generator(device=device.type).manual_seed(config.seed + 2)
            if device.type in {"cpu", "cuda"}
            else None
        )
        history: list[ChatTurn] = []
        print("注意：这是基础语言模型的对话格式补全，不是经过指令微调的可靠聊天助手。")
        print("输入 /reset 清空历史，输入 /quit 退出。")
        while True:
            try:
                user_message = input("你：").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_message:
                continue
            if user_message == "/quit":
                break
            if user_message == "/reset":
                history.clear()
                print("历史已清空。")
                continue
            response = generate_chat_response(
                model,
                tokenizer,
                system_prompt=args.system_prompt,
                history=history,
                user_message=user_message,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                generator=generator,
            )
            print(f"模型：{response}")
            history.append(ChatTurn(user=user_message, assistant=response))
        return 0
    if args.command == "benchmark":
        config = ProjectConfig.from_toml(args.config)
        output_path = config.artifact_dir / "attention_benchmark.json"
        print_json(
            benchmark_attention_implementations(
                dataclasses.replace(config.model, vocab_size=config.tokenizer.vocab_size),
                batch_size=args.batch_size or config.training.batch_size,
                seq_len=args.seq_len or config.training.seq_len,
                device_name=config.training.device,
                warmup=args.warmup,
                iterations=args.iterations,
                output_path=output_path,
            )
        )
        return 0
    if args.command == "benchmark-system":
        config = ProjectConfig.from_toml(args.config)
        output_path = config.artifact_dir / "system_benchmark.json"
        print_json(
            benchmark_modern_training_stack(
                dataclasses.replace(config.model, vocab_size=config.tokenizer.vocab_size),
                batch_size=args.batch_size or config.training.batch_size,
                seq_len=args.seq_len or config.training.seq_len,
                device_name=config.training.device,
                precision=config.training.precision,
                compile_model=config.training.compile_model,
                activation_checkpointing=config.training.activation_checkpointing,
                warmup=args.warmup,
                iterations=args.iterations,
                output_path=output_path,
            )
        )
        return 0
    if args.command == "fit-scaling":
        print_json(fit_from_json(args.runs, args.target_compute, args.output))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
