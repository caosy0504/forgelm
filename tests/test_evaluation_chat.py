from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from forgelm.chat import ChatTurn, build_chat_prompt, generate_chat_response
from forgelm.config import ModelConfig, TrainingConfig
from forgelm.data_pipeline import Document, write_jsonl
from forgelm.evaluation import evaluate_checkpoint
from forgelm.model import TransformerLM
from forgelm.tokenizer import BytePairTokenizer
from forgelm.training import Trainer, encode_documents


class EvaluationAndChatTests(unittest.TestCase):
    def test_chat_prompt_preserves_turn_order(self) -> None:
        prompt = build_chat_prompt(
            "Be helpful.",
            [ChatTurn(user="first question", assistant="first answer")],
            "second question",
        )
        self.assertLess(prompt.index("first question"), prompt.index("first answer"))
        self.assertLess(prompt.index("first answer"), prompt.index("second question"))
        self.assertTrue(prompt.endswith("Assistant:"))

    def test_checkpoint_evaluation_and_generation_paths(self) -> None:
        documents = [
            Document("a", "a clear experiment records measurements and limitations", "test", {}),
            Document("b", "a careful report explains each result with evidence", "test", {}),
        ]
        tokenizer = BytePairTokenizer.train((document.text for document in documents), vocab_size=270)
        tokens = encode_documents((document.text for document in documents), tokenizer)
        model_config = ModelConfig(
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_ff=32,
            max_seq_len=8,
            vocab_size=tokenizer.vocab_size,
        )
        training_config = TrainingConfig(
            steps=1,
            batch_size=1,
            seq_len=4,
            warmup_steps=0,
            eval_interval=1,
            eval_batches=1,
            checkpoint_interval=1,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_path = root / "tokenizer.json"
            validation_path = root / "validation.jsonl"
            tokenizer.save(tokenizer_path)
            write_jsonl(validation_path, documents)
            trainer = Trainer(
                TransformerLM(model_config),
                tokens,
                tokens,
                training_config,
                seed=9,
                artifact_dir=root,
            )
            checkpoint = trainer.train()["checkpoint"]
            report = evaluate_checkpoint(
                checkpoint_path=checkpoint,
                tokenizer_path=tokenizer_path,
                validation_jsonl=validation_path,
                device_name="cpu",
                batch_size=1,
                seq_len=4,
                eval_batches=1,
                seed=10,
                precision="fp32",
            )
            self.assertGreater(report["perplexity"], 0)
            response = generate_chat_response(
                trainer.model.eval(),
                tokenizer,
                system_prompt="Continue the conversation.",
                history=[],
                user_message="hello",
                device=torch.device("cpu"),
                max_new_tokens=2,
                temperature=1.0,
                top_k=0,
                generator=torch.Generator().manual_seed(2),
            )
            self.assertIsInstance(response, str)


if __name__ == "__main__":
    unittest.main()

