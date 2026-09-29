#!/usr/bin/env python3
"""Compare longitudinal beta-binomial M0 and lagged-exposure M1."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit, logsumexp

from src.models.measurement.dispersion_comparison import logpmf
from src.models.measurement.hierarchical_measurement import (
    FitResult,
    levels,
    posterior_summary,
    values,
)


FORMULAS = {
    "M0": (
        "proportion(successful, evaluated) ~ 1 + difficulty_z + day_z "
        "+ (1|exercise_id) + (1 + day_z|user_id)"
    ),
    "M1": (
        "proportion(successful, evaluated) ~ 1 + difficulty_z + day_z "
        "+ between_exposure + within_exposure "
        "+ (1|exercise_id) + (1 + day_z|user_id)"
    ),
}


def build_daily_exposure(events: pd.DataFrame) -> pd.DataFrame:
    """Build lagged exposure on a complete user-day grid for days 0--30."""
    users = sorted(events.user_id.astype(str).unique())
    grid = pd.MultiIndex.from_product(
        [users, range(31)], names=["user_id", "day_index"]
    ).to_frame(index=False)
    daily = (
        events.assign(user_id=events.user_id.astype(str))
        .loc[lambda frame: frame.day_index.between(0, 30)]
        .groupby(["user_id", "day_index"], observed=True).total_evaluated.sum()
        .rename("evaluated_today")
        .reset_index()
    )
    grid = grid.merge(daily, on=["user_id", "day_index"], how="left")
    grid["evaluated_today"] = grid.evaluated_today.fillna(0).astype(float)
    grid["exposure_raw"] = np.log1p(
        grid.groupby("user_id", observed=True).evaluated_today.shift(1).fillna(0)
    )
    train_mean = (
        grid.loc[grid.day_index.between(0, 20)]
        .groupby("user_id", observed=True).exposure_raw.mean()
        .rename("between_exposure")
    )
    grid = grid.join(train_mean, on="user_id")
    grid["within_exposure"] = grid.exposure_raw - grid.between_exposure
    return grid[[
        "user_id", "day_index", "exposure_raw", "between_exposure", "within_exposure"
    ]]


def prepare(
    modeling: pd.DataFrame, exposure: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    frame = modeling.copy()
    for column in ("user_id", "exercise_id", "response_type"):
        frame[column] = frame[column].astype(str)
    frame = frame.merge(
        exposure, on=["user_id", "day_index"], how="left", validate="many_to_one"
    )
    if frame[["exposure_raw", "between_exposure", "within_exposure"]].isna().any().any():
        raise ValueError("Missing lagged exposure after student-day join")
    train = frame.loc[frame.day_index.between(0, 20)].copy()
    test = frame.loc[frame.day_index.between(21, 30)].copy()
    scaling: dict[str, float] = {}
    for source, target in (("difficulty_level", "difficulty_z"), ("day_index", "day_z")):
        mean = float(train[source].mean())
        std = float(train[source].std(ddof=0))
        if not np.isfinite(std) or std == 0:
            std = 1.0
        train[target] = (train[source].astype(float) - mean) / std
        test[target] = (test[source].astype(float) - mean) / std
        scaling[f"{source}_mean"] = mean
        scaling[f"{source}_std"] = std
    return train, test, scaling


def priors(model_name: str) -> dict[str, bmb.Prior]:
    output = {
        "Intercept": bmb.Prior("Normal", mu=0, sigma=1.5),
        "difficulty_z": bmb.Prior("Normal", mu=0, sigma=1),
        "day_z": bmb.Prior("Normal", mu=0, sigma=0.5),
        "1|exercise_id": bmb.Prior(
            "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1)
        ),
        "1|user_id": bmb.Prior(
            "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1)
        ),
        "day_z|user_id": bmb.Prior(
            "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=0.5)
        ),
        "kappa": bmb.Prior("HalfNormal", sigma=50),
    }
    if model_name == "M1":
        output["between_exposure"] = bmb.Prior("Normal", mu=0, sigma=1)
        output["within_exposure"] = bmb.Prior("Normal", mu=0, sigma=1)
    return output


def fit(
    data: pd.DataFrame, model_name: str, iterations: int, draws: int, seed: int
) -> FitResult:
    model = bmb.Model(
        FORMULAS[model_name], data, family="beta_binomial",
        priors=priors(model_name), categorical=["exercise_id", "user_id"],
    )
    approximation = model.fit(
        inference_method="vi", n=iterations, method="advi",
        obj_optimizer=pm.adam(learning_rate=0.01), progressbar=False,
        random_seed=seed,
    )
    trace = approximation.sample(draws, return_inferencedata=False, random_seed=seed)
    return FitResult(model, trace, float(approximation.hist[-1]))


def probability_draws(
    result: FitResult,
    data: pd.DataFrame,
    model_name: str,
    seed: int,
    chunk_size: int = 5000,
):
    intercept = values(result.trace, "Intercept")
    fixed_names = ["difficulty_z", "day_z"]
    if model_name == "M1":
        fixed_names.extend(["between_exposure", "within_exposure"])
    fixed = {name: values(result.trace, name) for name in fixed_names}
    exercise_effect = values(result.trace, "1|exercise_id")
    user_effect = values(result.trace, "1|user_id")
    slope_effect = values(result.trace, "day_z|user_id")
    exercise_lookup = {name: index for index, name in enumerate(levels(result.model, "exercise_id"))}
    user_lookup = {name: index for index, name in enumerate(levels(result.model, "user_id"))}
    new_exercises = sorted(set(data.exercise_id) - set(exercise_lookup))
    new_users = sorted(set(data.user_id) - set(user_lookup))
    new_exercise_lookup = {name: index for index, name in enumerate(new_exercises)}
    new_user_lookup = {name: index for index, name in enumerate(new_users)}
    rng = np.random.default_rng(seed)
    sampled_exercise = rng.normal(size=(len(intercept), len(new_exercises))) * values(
        result.trace, "1|exercise_id_sigma"
    )[:, None]
    sampled_user = rng.normal(size=(len(intercept), len(new_users))) * values(
        result.trace, "1|user_id_sigma"
    )[:, None]
    sampled_slope = rng.normal(size=(len(intercept), len(new_users))) * values(
        result.trace, "day_z|user_id_sigma"
    )[:, None]

    for start in range(0, len(data), chunk_size):
        frame = data.iloc[start:start + chunk_size]
        eta = np.broadcast_to(intercept[:, None], (len(intercept), len(frame))).copy()
        for name, coefficient in fixed.items():
            eta += coefficient[:, None] * frame[name].to_numpy(float)[None, :]
        for position, exercise_id in enumerate(frame.exercise_id):
            if exercise_id in exercise_lookup:
                eta[:, position] += exercise_effect[:, exercise_lookup[exercise_id]]
            else:
                eta[:, position] += sampled_exercise[:, new_exercise_lookup[exercise_id]]
        day_z = frame.day_z.to_numpy(float)
        for position, user_id in enumerate(frame.user_id):
            if user_id in user_lookup:
                index = user_lookup[user_id]
                eta[:, position] += user_effect[:, index]
                eta[:, position] += slope_effect[:, index] * day_z[position]
            else:
                index = new_user_lookup[user_id]
                eta[:, position] += sampled_user[:, index]
                eta[:, position] += sampled_slope[:, index] * day_z[position]
        yield start, expit(eta)


def calibration(data: pd.DataFrame, probability: np.ndarray) -> tuple[pd.DataFrame, float]:
    frame = data[["day_index", "successful", "evaluated"]].copy()
    frame["predicted_successful"] = probability * frame.evaluated.to_numpy(float)
    daily = frame.groupby("day_index", observed=True).agg(
        rows=("successful", "size"), successful=("successful", "sum"),
        evaluated=("evaluated", "sum"),
        predicted_successful=("predicted_successful", "sum"),
    ).reset_index()
    daily["predicted_probability"] = daily.predicted_successful / daily.evaluated
    daily["observed_rate"] = daily.successful / daily.evaluated
    daily["absolute_calibration_error"] = (
        daily.predicted_probability - daily.observed_rate
    ).abs()
    error = float(np.average(daily.absolute_calibration_error, weights=daily.evaluated))
    return daily, error


def evaluate(
    result: FitResult, data: pd.DataFrame, model_name: str, seed: int
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    successful = data.successful.to_numpy(float)
    evaluated = data.evaluated.to_numpy(float)
    failures = evaluated - successful
    kappa = values(result.trace, "kappa")
    probability_mean = np.empty(len(data))
    user_names = sorted(data.user_id.unique())
    user_lookup = {name: index for index, name in enumerate(user_names)}
    user_codes = data.user_id.map(user_lookup).to_numpy()
    user_log_likelihood = np.zeros((len(kappa), len(user_names)))
    for start, probability in probability_draws(result, data, model_name, seed):
        stop = start + probability.shape[1]
        probability_mean[start:stop] = probability.mean(axis=0)
        ll = logpmf(
            "beta_binomial", successful[None, start:stop], evaluated[None, start:stop],
            probability, kappa,
        )
        for draw_index in range(len(probability)):
            np.add.at(
                user_log_likelihood[draw_index], user_codes[start:stop], ll[draw_index]
            )
    probability_mean = np.clip(probability_mean, 1e-9, 1 - 1e-9)
    daily, temporal_calibration = calibration(data, probability_mean)
    conditional_variance = (
        evaluated * probability_mean * (1 - probability_mean)
        * (evaluated + kappa.mean()) / (kappa.mean() + 1)
    )
    residual = (successful - evaluated * probability_mean) / np.sqrt(
        np.maximum(conditional_variance, 1e-9)
    )
    metrics = {
        "rows": int(len(data)), "students": int(data.user_id.nunique()),
        "evaluated_elements": int(evaluated.sum()),
        "joint_user_lpd": float(np.sum(
            logsumexp(user_log_likelihood, axis=0) - np.log(len(kappa))
        )),
        "binary_log_loss": float(-np.sum(
            successful * np.log(probability_mean)
            + failures * np.log1p(-probability_mean)
        ) / evaluated.sum()),
        "brier_score": float(np.sum(
            successful * (1 - probability_mean) ** 2
            + failures * probability_mean**2
        ) / evaluated.sum()),
        "temporal_calibration_error": temporal_calibration,
        "pearson_residual_mean": float(residual.mean()),
        "pearson_residual_sd": float(residual.std(ddof=1)),
        "kappa": float(kappa.mean()),
    }
    predictions = data[[
        "user_id", "exercise_id", "day_index", "successful", "evaluated",
        "exposure_raw", "between_exposure", "within_exposure",
    ]].copy()
    predictions["predicted_probability"] = probability_mean
    predictions["observed_rate"] = successful / evaluated
    predictions["pearson_residual"] = residual
    return metrics, predictions, daily


def parameter_summary(
    result: FitResult,
    model_name: str,
    response_type: str,
    day_std: float,
) -> pd.DataFrame:
    random_slope = values(result.trace, "day_z|user_id")
    beta_time = (
        values(result.trace, "day_z") + random_slope.mean(axis=1)
    ) / day_std
    sigma_slope = values(result.trace, "day_z|user_id_sigma") / day_std
    draws = {
        "beta_time_per_day": beta_time,
        "sigma_student_slope_per_day": sigma_slope,
    }
    if model_name == "M1":
        draws["beta_between_exposure"] = values(result.trace, "between_exposure")
        draws["beta_within_exposure"] = values(result.trace, "within_exposure")
    rows = []
    for parameter, samples in draws.items():
        summary = posterior_summary(samples[:, None])
        rows.append({
            "response_type": response_type, "model": model_name,
            "parameter": parameter,
            **{name: float(value[0]) for name, value in summary.items()},
            "probability_positive": float((samples > 0).mean()),
            "interval_excludes_zero": bool(
                summary["hdi_3pct"][0] > 0 or summary["hdi_97pct"][0] < 0
            ),
        })
    return pd.DataFrame(rows)


def markdown(frame: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_report(
    comparison: pd.DataFrame,
    parameters: pd.DataFrame,
    temporal: pd.DataFrame,
    metadata: dict,
) -> str:
    metrics_view = comparison[[
        "evaluation", "response_type", "model", "joint_user_lpd",
        "delta_lpd_vs_M0", "binary_log_loss", "delta_log_loss_vs_M0",
        "brier_score", "temporal_calibration_error",
    ]]
    parameter_view = parameters[[
        "response_type", "model", "parameter", "estimate", "posterior_sd",
        "hdi_3pct", "hdi_97pct", "probability_positive", "interval_excludes_zero",
    ]]
    daily_view = temporal[[
        "response_type", "model", "day_index", "evaluated",
        "predicted_probability", "observed_rate", "absolute_calibration_error",
    ]]
    return f"""# Lagged exposure and longitudinal platform performance

## Executive summary

This experiment compares only M0 and M1. It asks whether prior-day evaluated
exposure is associated with subsequent platform-recorded performance after
adjusting for student level, exercise, declared difficulty, population time,
and student-specific temporal slopes. Results are associational, not causal.

M0 supports a positive average temporal trend and non-zero heterogeneity in
student slopes. In M1, both between-student and within-student exposure
coefficients are positive and their approximate ADVI intervals exclude zero.
However, M1 performs materially worse than M0 in both out-of-sample designs and
systematically overpredicts success. The exposure association estimated in the
training period therefore does not generalize as a stable predictive association
to days 21--30.

## Design

- Training: `[0, 21)`.
- Primary evaluation: rolling-origin `[21, 31)`; the prediction for day `t` uses
  observed exposure from day `t-1` and no information from day `t` or later.
- Sensitivity: frozen day 21; outcomes on day 21 use exposure from day 20.
- The student-specific exposure mean is calculated only over training calendar
  days 0--20, including zero-exposure days, and remains frozen in both evaluations.
- Notes and chords are modeled separately with hierarchical beta-binomial models.
- Inference: {metadata['vi_iterations']:,} ADVI iterations and
  {metadata['posterior_draws']:,} posterior draws.

Exposure is defined as:

$$ExposureRaw_{{i,t-1}}=\\log(1+EvaluatedElements_{{i,t-1}})$$

$$BetweenExposure_i=mean_{{t\\in[0,21)}}(ExposureRaw_{{i,t-1}})$$

$$WithinExposure_{{i,t}}=ExposureRaw_{{i,t-1}}-BetweenExposure_i$$

## Out-of-sample comparison

{markdown(metrics_view)}

Higher LPD and lower log-loss, Brier score, and temporal calibration error are
better. M1 should be considered an improvement only if gains are reasonably
consistent across these criteria and response types.

## Posterior parameters fitted on `[0, 21)`

{markdown(parameter_view)}

`beta_time_per_day` and `sigma_student_slope_per_day` are expressed in log-odds
per elapsed day. Exposure coefficients represent a one-unit change in lagged
`log1p(evaluated elements)`. An interval excluding zero is evidence of posterior
identifiability under mean-field ADVI, not evidence of a causal effect.

Adding exposure does not substantially reorganize the baseline temporal
structure. For notes, `beta_time_per_day` changes from 0.0485 to 0.0511 (+5.3%)
and `sigma_student_slope_per_day` from 0.0402 to 0.0396 (-1.4%). For chords,
the corresponding changes are 0.0390 to 0.0395 (+1.4%) and 0.0407 to 0.0421
(+3.4%).

## Rolling-origin calibration by day

{markdown(daily_view)}

## Interpretation boundaries

M1 is not retained as an improvement over M0. In rolling-origin evaluation it
worsens joint LPD by 1,932 for notes and 1,268 for chords, increases log-loss,
and increases temporal calibration error from 0.0092 to 0.0650 for notes and
from 0.0231 to 0.0472 for chords. Frozen day-21 sensitivity leads to the same
decision. Positive training-period coefficients alone are insufficient evidence
to proceed to an exposure-by-time interaction.

Rolling-origin is a dynamic longitudinal evaluation, not a frozen intervention
model at day 21. For outcomes after day 21 it uses only exposure already observed
on the immediately preceding day. Frozen day-21 sensitivity answers the narrower
question using information available at the original cutoff.

The exposure coefficients describe conditional associations. Motivation,
previous musical experience, available time, and other unrecorded factors can
affect both practice and performance, so the estimates must not be interpreted
as effects of increasing practice.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events", type=Path, default=Path("data/interim/events_clean.parquet")
    )
    parser.add_argument(
        "--modeling-table", type=Path,
        default=Path("data/processed/student_exercise_day.parquet"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/model_outputs/longitudinal_exposure_v1"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("reports/measurement/longitudinal_exposure_v1.md"),
    )
    parser.add_argument("--vi-iterations", type=int, default=20000)
    parser.add_argument("--posterior-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260929)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(args.events)
    modeling = pd.read_parquet(args.modeling_table)
    exposure = build_daily_exposure(events)
    train_all, test_all, scaling = prepare(modeling, exposure)
    exposure.to_parquet(args.output_dir / "student_day_lagged_exposure.parquet", index=False)

    metric_rows, parameter_frames = [], []
    prediction_frames, calibration_frames = [], []
    for response_index, response_type in enumerate(("note", "chord")):
        train = train_all.loc[train_all.response_type == response_type].copy()
        rolling = test_all.loc[test_all.response_type == response_type].copy()
        frozen = rolling.loc[rolling.day_index == 21].copy()
        for model_index, model_name in enumerate(("M0", "M1")):
            print(f"Fitting {response_type} {model_name}", flush=True)
            result = fit(
                train, model_name, args.vi_iterations, args.posterior_draws,
                args.seed + response_index * 10 + model_index,
            )
            parameter_frames.append(parameter_summary(
                result, model_name, response_type, scaling["day_index_std"]
            ))
            for evaluation_index, (evaluation_name, evaluation_data) in enumerate((
                ("rolling_21_31", rolling), ("frozen_day_21", frozen)
            )):
                metrics, predictions, daily = evaluate(
                    result, evaluation_data, model_name,
                    args.seed + 100 + response_index * 10 + model_index * 2 + evaluation_index,
                )
                metric_rows.append({
                    "evaluation": evaluation_name, "response_type": response_type,
                    "model": model_name, **metrics,
                })
                predictions.insert(0, "evaluation", evaluation_name)
                predictions.insert(1, "response_type", response_type)
                predictions.insert(2, "model", model_name)
                prediction_frames.append(predictions)
                daily.insert(0, "evaluation", evaluation_name)
                daily.insert(1, "response_type", response_type)
                daily.insert(2, "model", model_name)
                calibration_frames.append(daily)
            del result
            gc.collect()

    comparison = pd.DataFrame(metric_rows)
    for metric, delta_name, direction in (
        ("joint_user_lpd", "delta_lpd_vs_M0", 1),
        ("binary_log_loss", "delta_log_loss_vs_M0", -1),
        ("brier_score", "delta_brier_vs_M0", -1),
    ):
        baseline = comparison.loc[comparison.model == "M0", [
            "evaluation", "response_type", metric
        ]].rename(columns={metric: "baseline"})
        comparison = comparison.merge(
            baseline, on=["evaluation", "response_type"], how="left", validate="many_to_one"
        )
        comparison[delta_name] = direction * (comparison[metric] - comparison.baseline)
        comparison = comparison.drop(columns="baseline")
    parameters = pd.concat(parameter_frames, ignore_index=True)
    temporal = pd.concat(calibration_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
    parameters.to_csv(args.output_dir / "posterior_parameters.csv", index=False)
    temporal.to_csv(args.output_dir / "temporal_calibration.csv", index=False)
    predictions.to_parquet(args.output_dir / "predictions.parquet", index=False)
    metadata = {
        "train_window": "[0, 21)", "rolling_evaluation": "[21, 31)",
        "frozen_evaluation": "day 21 only", "exposure_lag": "one calendar day",
        "between_exposure_window": "training days 0-20 only, including zero days",
        "train_rows": len(train_all), "rolling_test_rows": len(test_all),
        "frozen_test_rows": int((test_all.day_index == 21).sum()),
        "vi_iterations": args.vi_iterations, "posterior_draws": args.posterior_draws,
        "seed": args.seed, "scaling": scaling,
        "causal_interpretation": False,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    report_temporal = temporal.loc[temporal.evaluation == "rolling_21_31"].copy()
    args.report.write_text(
        render_report(comparison, parameters, report_temporal, metadata), encoding="utf-8"
    )
    print(comparison.to_string(index=False))
    print(parameters.to_string(index=False))


if __name__ == "__main__":
    main()
