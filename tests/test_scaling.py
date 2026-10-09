from __future__ import annotations

import unittest

from forgelm.scaling import fit_isoflops


class ScalingTests(unittest.TestCase):
    def test_recovers_synthetic_half_power_laws(self) -> None:
        records: list[dict[str, float]] = []
        for compute in (1e6, 1e8, 1e10, 1e12):
            optimum = compute**0.5
            records.extend(
                [
                    {"compute_budget": compute, "parameters": optimum / 2, "final_loss": 2.1},
                    {"compute_budget": compute, "parameters": optimum, "final_loss": 2.0},
                    {"compute_budget": compute, "parameters": optimum * 2, "final_loss": 2.2},
                ]
            )
        fit = fit_isoflops(records)
        self.assertAlmostEqual(fit.parameter_exponent, 0.5, places=6)
        self.assertAlmostEqual(fit.token_exponent, 0.5, places=6)
        prediction = fit.predict(1e14)
        self.assertAlmostEqual(prediction["predicted_parameters"], 1e7, delta=1.0)


if __name__ == "__main__":
    unittest.main()

