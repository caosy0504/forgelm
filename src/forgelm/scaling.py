from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ScalingFit:
    parameter_coefficient: float
    parameter_exponent: float
    token_coefficient: float
    token_exponent: float
    selected_points: list[dict[str, float]]
    parameter_log_rmse: float
    token_log_rmse: float

    def predict(self, compute_budget: float) -> dict[str, float]:
        if compute_budget <= 0:
            raise ValueError("compute_budget must be positive")
        parameters = self.parameter_coefficient * compute_budget**self.parameter_exponent
        tokens = self.token_coefficient * compute_budget**self.token_exponent
        return {
            "compute_budget": compute_budget,
            "predicted_parameters": parameters,
            "predicted_tokens": tokens,
            "implied_compute_6nd": 6.0 * parameters * tokens,
        }


def _fit_power_law(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float, float]:
    if len(x_values) < 2:
        raise ValueError("at least two IsoFLOPs budgets are required")
    slope, intercept = np.polyfit(np.log(x_values), np.log(y_values), deg=1)
    predictions = intercept + slope * np.log(x_values)
    rmse = float(np.sqrt(np.mean((np.log(y_values) - predictions) ** 2)))
    return float(math.exp(intercept)), float(slope), rmse


def fit_isoflops(records: Iterable[dict[str, float]]) -> ScalingFit:
    grouped: dict[float, list[dict[str, float]]] = {}
    for record in records:
        compute = float(record["compute_budget"])
        parameters = float(record["parameters"])
        loss = float(record["final_loss"])
        if min(compute, parameters, loss) <= 0:
            raise ValueError("compute_budget, parameters, and final_loss must be positive")
        grouped.setdefault(compute, []).append({"compute_budget": compute, "parameters": parameters, "final_loss": loss})

    selected: list[dict[str, float]] = []
    for compute, candidates in sorted(grouped.items()):
        best = min(candidates, key=lambda item: (item["final_loss"], item["parameters"]))
        selected.append(
            {
                **best,
                "tokens": compute / (6.0 * best["parameters"]),
            }
        )
    compute_values = np.array([point["compute_budget"] for point in selected], dtype=np.float64)
    parameter_values = np.array([point["parameters"] for point in selected], dtype=np.float64)
    token_values = np.array([point["tokens"] for point in selected], dtype=np.float64)
    parameter_coefficient, parameter_exponent, parameter_rmse = _fit_power_law(compute_values, parameter_values)
    token_coefficient, token_exponent, token_rmse = _fit_power_law(compute_values, token_values)
    return ScalingFit(
        parameter_coefficient=parameter_coefficient,
        parameter_exponent=parameter_exponent,
        token_coefficient=token_coefficient,
        token_exponent=token_exponent,
        selected_points=selected,
        parameter_log_rmse=parameter_rmse,
        token_log_rmse=token_rmse,
    )


def fit_from_json(path: str | Path, target_compute: float, output_path: str | Path | None = None) -> dict[str, object]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    fit = fit_isoflops(records)
    result: dict[str, object] = {"fit": asdict(fit), "prediction": fit.predict(target_compute)}
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result

