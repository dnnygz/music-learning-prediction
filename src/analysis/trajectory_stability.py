"""Exploratory estimators for future raw-performance trajectory stability."""

from __future__ import annotations

import numpy as np
import pandas as pd


WINDOWS = {
    "days_7_14": (7.0, 14.0),
    "days_7_21": (7.0, 21.0),
    "days_7_31": (7.0, 31.0),
}


def aggregate_daily_performance(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate successes and denominators before calculating daily accuracy."""
    future = events.loc[
        events["days_since_signup"].between(7.0, 31.0, inclusive="left")
        & (events["total_evaluated"] > 0)
    ]
    daily = (
        future.groupby(["user_id", "day_index"], observed=True, as_index=False)
        .agg(
            successful=("total_successful", "sum"),
            evaluated=("total_evaluated", "sum"),
            events=("source_row_number", "size"),
        )
        .sort_values(["user_id", "day_index"])
    )
    daily["accuracy"] = daily["successful"] / daily["evaluated"]
    return daily


def ordinary_slope(group: pd.DataFrame) -> pd.Series:
    """Estimate an unweighted daily-accuracy slope and classical standard error."""
    x = group["day_index"].to_numpy(dtype=float)
    y = group["accuracy"].to_numpy(dtype=float)
    x_centered = x - x.mean()
    sum_squares_x = float(np.dot(x_centered, x_centered))
    if len(x) < 3 or sum_squares_x == 0:
        return pd.Series({"slope": np.nan, "slope_se": np.nan, "r_squared": np.nan})

    slope = float(np.dot(x_centered, y - y.mean()) / sum_squares_x)
    intercept = float(y.mean() - slope * x.mean())
    residuals = y - (intercept + slope * x)
    residual_sum_squares = float(np.dot(residuals, residuals))
    total_sum_squares = float(np.dot(y - y.mean(), y - y.mean()))
    residual_variance = residual_sum_squares / (len(x) - 2)
    slope_se = float(np.sqrt(residual_variance / sum_squares_x))
    r_squared = 1.0 - residual_sum_squares / total_sum_squares if total_sum_squares > 0 else np.nan
    return pd.Series({"slope": slope, "slope_se": slope_se, "r_squared": r_squared})


def estimate_window_slopes(
    daily: pd.DataFrame,
    start: float,
    end: float,
    min_active_days: int = 3,
) -> pd.DataFrame:
    """Estimate one raw daily-accuracy slope per sufficiently observed student."""
    window = daily.loc[daily["day_index"].between(start, end, inclusive="left")]
    summary = window.groupby("user_id", observed=True).agg(
        active_days=("day_index", "nunique"),
        evaluated=("evaluated", "sum"),
        first_day=("day_index", "min"),
        last_day=("day_index", "max"),
    )
    eligible_users = summary.index[summary["active_days"] >= min_active_days]
    estimates = (
        window.loc[window["user_id"].isin(eligible_users)]
        .groupby("user_id", observed=True)
        .apply(ordinary_slope, include_groups=False)
    )
    result = summary.loc[eligible_users].join(estimates)
    result["slope_ci_low"] = result["slope"] - 1.96 * result["slope_se"]
    result["slope_ci_high"] = result["slope"] + 1.96 * result["slope_se"]
    result["sign_resolved"] = (result["slope_ci_low"] > 0) | (result["slope_ci_high"] < 0)
    return result.reset_index()


def estimate_all_windows(daily: pd.DataFrame, min_active_days: int = 3) -> dict[str, pd.DataFrame]:
    """Estimate comparable slopes for the predefined nested future windows."""
    return {
        name: estimate_window_slopes(daily, start, end, min_active_days=min_active_days)
        for name, (start, end) in WINDOWS.items()
    }


def compare_to_full_window(estimates: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare truncated-window slopes with the full `[7, 31)` diagnostic slope."""
    full = estimates["days_7_31"][["user_id", "slope"]].rename(
        columns={"slope": "full_slope"}
    )
    rows = []
    for name, frame in estimates.items():
        merged = frame[["user_id", "slope"]].merge(full, on="user_id", how="inner")
        rows.append(
            {
                "window": name,
                "students": len(merged),
                "pearson": merged["slope"].corr(merged["full_slope"], method="pearson"),
                "spearman": merged["slope"].rank().corr(merged["full_slope"].rank()),
                "sign_agreement": float(
                    (np.sign(merged["slope"]) == np.sign(merged["full_slope"])).mean()
                ),
                "mae_vs_full": float((merged["slope"] - merged["full_slope"]).abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def stability_by_minimum_days(daily: pd.DataFrame) -> pd.DataFrame:
    """Summarize full-window slope uncertainty under alternative coverage cutoffs."""
    rows = []
    for minimum_days in (3, 4, 5, 7, 10, 14):
        estimates = estimate_window_slopes(daily, 7.0, 31.0, min_active_days=minimum_days)
        rows.append(
            {
                "min_active_days": minimum_days,
                "students": len(estimates),
                "median_slope_se": estimates["slope_se"].median(),
                "median_ci_width": (estimates["slope_ci_high"] - estimates["slope_ci_low"]).median(),
                "resolved_sign_rate": estimates["sign_resolved"].mean(),
                "median_r_squared": estimates["r_squared"].median(),
            }
        )
    return pd.DataFrame(rows)
