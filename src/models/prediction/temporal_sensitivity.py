#!/usr/bin/env python3
"""Compare 7-, 14-, and 21-day histories on one common future trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from src.data.build_prediction_dataset import build_features
from src.models.prediction.two_stage_prediction import run


HISTORY_WINDOWS = (7, 14, 21)


def eligible_users(events: pd.DataFrame, end: int) -> set[str]:
    window = events.loc[events.days_since_signup.between(0, end, inclusive="left")]
    summary = window.groupby("user_id", observed=True).agg(
        active_days=("day_index", "nunique"),
        evaluated=("total_evaluated", "sum"),
    )
    return set(summary.index[(summary.active_days >= 1) & (summary.evaluated >= 1)])


def future_observable_users(events: pd.DataFrame) -> set[str]:
    future = events.loc[events.days_since_signup.between(21, 31, inclusive="left")].copy()
    future["evaluated_day"] = future.day_index.where(future.total_evaluated > 0)
    summary = future.groupby("user_id", observed=True).agg(
        evaluated_days=("evaluated_day", "nunique"),
        evaluated=("total_evaluated", "sum"),
    )
    return set(summary.index[(summary.evaluated_days >= 3) & (summary.evaluated >= 100)])


def add_targets(features: pd.DataFrame, slopes: pd.DataFrame) -> pd.DataFrame:
    targets = slopes.loc[slopes.window == "21_31"].copy()
    targets["calibrated_slope_sd"] = targets.slope_sd
    targets["probability_slope_gt_0p02"] = ndtr(
        (targets.adjusted_trajectory_slope - 0.02)
        / targets.calibrated_slope_sd.clip(lower=1e-9)
    )
    return features.merge(targets, on="user_id", how="inner", validate="one_to_many")


def constant_baselines(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for response_type, frame in data.groupby("response_type", observed=True):
        q = frame.probability_slope_gt_0p02.to_numpy()
        slope = frame.adjusted_trajectory_slope.to_numpy()
        constant_q = float(q.mean())
        constant_slope = float(slope.mean())
        rows.extend([
            {
                "formulation": "A_probability", "response_type": response_type,
                "model": "constant_baseline",
                "expected_log_loss": float(np.mean(
                    -q * np.log(constant_q) - (1 - q) * np.log(1 - constant_q)
                )),
                "expected_brier": float(np.mean(
                    q * (1 - constant_q) ** 2 + (1 - q) * constant_q**2
                )),
                "mae_probability": float(np.mean(np.abs(q - constant_q))),
                "correlation_probability": np.nan,
            },
            {
                "formulation": "B_continuous_multiple_imputation",
                "response_type": response_type, "model": "constant_baseline",
                "rmse_posterior_mean": float(np.sqrt(np.mean((slope - constant_slope) ** 2))),
                "mae_posterior_mean": float(np.mean(np.abs(slope - constant_slope))),
                "r2_posterior_mean": 0.0,
                "correlation_posterior_mean": np.nan,
                "expected_rmse_including_target_uncertainty": float(np.sqrt(np.mean(
                    (slope - constant_slope) ** 2 + frame.calibrated_slope_sd.to_numpy() ** 2
                ))),
            },
        ])
    return pd.DataFrame(rows)


def render_report(
    cohort: pd.DataFrame, metrics: pd.DataFrame, metadata: dict[str, object]
) -> str:
    def table(frame: pd.DataFrame) -> str:
        lines = [
            "| " + " | ".join(frame.columns) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |",
        ]
        for row in frame.itertuples(index=False, name=None):
            values = [
                f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
                for value in row
            ]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    continuous = metrics.loc[
        metrics.formulation == "B_continuous_multiple_imputation",
        ["history_days", "response_type", "model", "rmse_posterior_mean",
         "r2_posterior_mean", "correlation_posterior_mean"],
    ]
    probability = metrics.loc[
        metrics.formulation == "A_probability",
        ["history_days", "response_type", "model", "expected_log_loss",
         "expected_brier", "correlation_probability"],
    ]
    return f"""# Temporal sensitivity of adjusted-trajectory prediction

## Question

Is the failure of early trajectory prediction caused by insufficient observation
time, or does the platform lack enough early behavioral signal to identify later
individual performance dynamics?

## Controlled design

- Histories compared: `[0, 7)`, `[0, 14)`, and `[0, 21)`.
- Common target window: `[21, 31)`.
- Initial eligibility: at least one active day and one evaluated element.
- Target observability: at least three evaluated days and 100 evaluated elements.
- Main comparison cohort: intersection of all initial cohorts and target-observable
  students.
- Common students: {metadata['common_analysis_students']}.
- Students with an estimable note target: {metadata['common_note_target_students']}.
- Students with an estimable chord target: {metadata['common_chord_target_students']}.
- Identical user folds are used for every history window.

The common future window prevents temporal overlap. The common cohort prevents a
longer-history model from appearing better merely because its population changed.

## Cohort accounting

{table(cohort)}

Eligibility increases with a longer observation window because the windows are
nested: a student eligible at day 7 remains eligible at days 14 and 21. The main
sample reduction comes from requiring a trajectory estimable in the shorter
future window `[21, 31)`, not from losing initial history.

## Continuous adjusted-slope prediction

{table(continuous)}

## Probability of a relevant positive slope

The operational probability target is $P(\\lambda_i > 0.02)$.

{table(probability)}

## Interpretation

Extending the initial history does not recover
useful predictive signal. Every continuous model has negative out-of-fold
$R^2$, so each performs worse than predicting the response-specific cohort mean.
All probabilistic models also have slightly worse expected log-loss than their
constant baselines. Correlations remain close to zero and do not improve
consistently from 7 to 14 or 21 days.

Therefore, under the current features, target, and common-cohort design, the
failure of trajectory prediction cannot be attributed only to using seven days
of initial observation. Waiting until day 14 or day 21 does not materially improve
identification of later individual adjusted-performance dynamics.

The `[21, 31)` slope target is estimated with mean-field ADVI. Its uncertainty is
propagated in the continuous two-stage models, but no window-specific NUTS
validation was performed; conclusions should emphasize relative predictive
performance rather than exact posterior coverage.

The fixed target contains only ten elapsed days, compared with 24 days in the
original `[7, 31)` analysis. This is necessary to compare all histories without
overlap, but can make the common trajectory noisier. The experiment answers
whether longer histories improve prediction of this same later target; it does
not prove that every possible longer-horizon trajectory target is unpredictable.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path("data/interim/events_clean.parquet"))
    parser.add_argument(
        "--slopes", type=Path,
        default=Path("data/model_outputs/temporal_sensitivity_target_v1/student_adjusted_trajectory_slopes.parquet"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/model_outputs/temporal_sensitivity_v1"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("reports/prediction/temporal_sensitivity_v1.md"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--imputations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260928)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(args.events)
    slopes = pd.read_parquet(args.slopes)
    initial_sets = {end: eligible_users(events, end) for end in HISTORY_WINDOWS}
    target_observable = future_observable_users(events)
    common_initial = set.intersection(*initial_sets.values())
    common_analysis = common_initial & target_observable
    target_users_by_response = {
        response_type: set(frame.user_id) & common_analysis
        for response_type, frame in slopes.groupby("response_type", observed=True)
    }

    cohort_rows = []
    for end in HISTORY_WINDOWS:
        cohort_rows.append({
            "history_days": end,
            "initial_eligible": len(initial_sets[end]),
            "target_observable": len(target_observable),
            "eligible_and_target_observable": len(initial_sets[end] & target_observable),
            "common_analysis_cohort": len(common_analysis),
        })
    cohort = pd.DataFrame(cohort_rows)

    metric_frames, prediction_frames = [], []
    baseline = None
    for end in HISTORY_WINDOWS:
        features = build_features(events, initial_end=end)
        dataset = add_targets(features, slopes)
        dataset = dataset.loc[dataset.user_id.isin(common_analysis)].copy()
        if baseline is None:
            baseline = constant_baselines(dataset)
            baseline.insert(0, "history_days", 0)
        metrics, predictions = run(dataset, args.folds, args.imputations, args.seed)
        metrics.insert(0, "history_days", end)
        predictions.insert(0, "history_days", end)
        metric_frames.append(metrics)
        prediction_frames.append(predictions)

    all_metrics = pd.concat([baseline, *metric_frames], ignore_index=True, sort=False)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    cohort.to_csv(args.output_dir / "cohort_counts.csv", index=False)
    all_metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    all_predictions.to_parquet(args.output_dir / "out_of_fold_predictions.parquet", index=False)
    metadata = {
        "history_windows": ["[0, 7)", "[0, 14)", "[0, 21)"],
        "target_window": "[21, 31)",
        "initial_eligibility": ">=1 active day and >=1 evaluated element",
        "target_observability": ">=3 evaluated days and >=100 evaluated elements",
        "common_initial_students": len(common_initial),
        "target_observable_students": len(target_observable),
        "common_analysis_students": len(common_analysis),
        "common_note_target_students": len(target_users_by_response.get("note", set())),
        "common_chord_target_students": len(target_users_by_response.get("chord", set())),
        "folds": args.folds, "imputations": args.imputations, "seed": args.seed,
        "delta": 0.02,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    args.report.write_text(render_report(cohort, all_metrics, metadata), encoding="utf-8")
    print(cohort.to_string(index=False))
    print(all_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
