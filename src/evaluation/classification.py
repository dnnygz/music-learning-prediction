"""Reusable evaluation helpers for binary probabilistic classifiers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def quantile_calibration(
    y: np.ndarray, probability: np.ndarray, bins: int = 5
) -> pd.DataFrame:
    """Summarize calibration in approximately equal-sized probability bins."""
    frame = pd.DataFrame({"outcome": y, "probability": probability})
    frame["bin"] = pd.qcut(
        frame.probability.rank(method="first"), q=bins, labels=False
    )
    return (
        frame.groupby("bin", observed=True)
        .agg(
            n=("outcome", "size"),
            predicted_probability_mean=("probability", "mean"),
            observed_rate=("outcome", "mean"),
        )
        .reset_index()
    )


def stratified_bootstrap_interval(
    y: np.ndarray,
    probability: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    """Return a percentile interval while preserving both outcome classes."""
    rng = np.random.default_rng(seed)
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    values = np.empty(draws)
    for draw in range(draws):
        sample = np.concatenate([
            rng.choice(negative, size=len(negative), replace=True),
            rng.choice(positive, size=len(positive), replace=True),
        ])
        values[draw] = metric(y[sample], probability[sample])
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)
