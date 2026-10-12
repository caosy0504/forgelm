"""Numerical checks for probability ensembling, including uneven confidence."""

import math
import unittest

import torch

from scripts.evaluate_ensemble import equal_probability_mixture, gold_log_probabilities


class ProbabilityEnsembleTests(unittest.TestCase):
    def test_mixes_probabilities_instead_of_logits(self) -> None:
        targets = torch.tensor([0])
        first_logits = torch.tensor([[math.log(9.0), 0.0]])
        second_logits = torch.tensor([[0.0, math.log(9.0)]])
        first = gold_log_probabilities(first_logits, targets)
        second = gold_log_probabilities(second_logits, targets)
        mixed = equal_probability_mixture(first, second)
        self.assertAlmostEqual(float(first.exp()), 0.9, places=6)
        self.assertAlmostEqual(float(second.exp()), 0.1, places=6)
        self.assertAlmostEqual(float(mixed.exp()), 0.5, places=6)

    def test_identical_models_leave_predictions_unchanged(self) -> None:
        scores = gold_log_probabilities(torch.tensor([[3.0, 1.0, -2.0]]), torch.tensor([2]))
        self.assertTrue(torch.allclose(equal_probability_mixture(scores, scores), scores))

    def test_rejects_mismatched_shapes(self) -> None:
        with self.assertRaises(ValueError):
            equal_probability_mixture(torch.zeros(2), torch.zeros(3))


if __name__ == "__main__":
    unittest.main()
