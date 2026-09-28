#!/usr/bin/env python3
"""Fit and validate beta-binomial student random temporal slopes."""

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
from scipy.stats import spearmanr

from src.models.dispersion_comparison import conditional_calibration, logpmf
from src.models.hierarchical_measurement import (
    RANDOM_SEED, FitResult, posterior_summary, prepare_data, split_users,
    standardize, values,
)


FORMULAS = {
    "random_intercept": (
        "proportion(successful, evaluated) ~ 1 + difficulty_z + day_z "
        "+ (1|exercise_id) + (1|user_id)"
    ),
    "random_slope": (
        "proportion(successful, evaluated) ~ 1 + difficulty_z + day_z "
        "+ (1|exercise_id) + (1 + day_z|user_id)"
    ),
}


def model_priors(model_name: str) -> dict[str, bmb.Prior]:
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
        "kappa": bmb.Prior("HalfNormal", sigma=50),
    }
    if model_name == "random_slope":
        output["day_z|user_id"] = bmb.Prior(
            "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=0.5)
        )
    return output


def fit_model(
    data: pd.DataFrame, model_name: str, iterations: int, draws: int, seed: int
) -> FitResult:
    model = bmb.Model(
        FORMULAS[model_name], data, family="beta_binomial",
        priors=model_priors(model_name), categorical=["exercise_id", "user_id"],
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
    difficulty = values(result.trace, "difficulty_z")
    global_time = values(result.trace, "day_z")
    exercise = values(result.trace, "1|exercise_id")
    exercise_levels = [str(x) for x in result.model.backend.model.coords["exercise_id__factor_dim"]]
    exercise_lookup = {name: index for index, name in enumerate(exercise_levels)}
    sigma_intercept = values(result.trace, "1|user_id_sigma")
    sigma_slope = (
        values(result.trace, "day_z|user_id_sigma")
        if model_name == "random_slope" else None
    )
    users = sorted(data.user_id.unique())
    user_lookup = {name: index for index, name in enumerate(users)}
    rng = np.random.default_rng(seed)
    new_intercept = rng.normal(size=(len(intercept), len(users))) * sigma_intercept[:, None]
    new_slope = (
        rng.normal(size=(len(intercept), len(users))) * sigma_slope[:, None]
        if sigma_slope is not None else None
    )
    for start in range(0, len(data), chunk_size):
        frame = data.iloc[start : start + chunk_size]
        difficulty_z = frame.difficulty_z.to_numpy()
        day_z = frame.day_z.to_numpy()
        eta = (
            intercept[:, None]
            + difficulty[:, None] * difficulty_z[None, :]
            + global_time[:, None] * day_z[None, :]
        )
        exercise_idx = np.array([exercise_lookup.get(x, -1) for x in frame.exercise_id])
        known = exercise_idx >= 0
        if known.any():
            eta[:, known] += exercise[:, exercise_idx[known]]
        user_idx = np.array([user_lookup[x] for x in frame.user_id])
        eta += new_intercept[:, user_idx]
        if new_slope is not None:
            eta += new_slope[:, user_idx] * day_z[None, :]
        yield start, expit(eta)


def evaluate(result: FitResult, data: pd.DataFrame, model_name: str, seed: int):
    successful = data.successful.to_numpy(float)
    evaluated = data.evaluated.to_numpy(float)
    kappa = values(result.trace, "kappa")
    p_mean = np.empty(len(data))
    predictive_variance = np.empty(len(data))
    users = sorted(data.user_id.unique())
    user_lookup = {name: index for index, name in enumerate(users)}
    codes = data.user_id.map(user_lookup).to_numpy()
    user_ll = None
    for start, probability in probability_draws(result, data, model_name, seed):
        stop = start + probability.shape[1]
        n = evaluated[None, start:stop]
        y = successful[None, start:stop]
        ll = logpmf("beta_binomial", y, n, probability, kappa)
        if user_ll is None:
            user_ll = np.zeros((len(probability), len(users)))
        for draw_index in range(len(probability)):
            np.add.at(user_ll[draw_index], codes[start:stop], ll[draw_index])
        conditional_mean = n * probability
        conditional_variance = (
            n * probability * (1 - probability) * (n + kappa[:, None])
            / (kappa[:, None] + 1)
        )
        p_mean[start:stop] = probability.mean(axis=0)
        predictive_variance[start:stop] = (
            conditional_variance.mean(axis=0) + conditional_mean.var(axis=0, ddof=1)
        )
    failures = evaluated - successful
    residual = (successful - evaluated * p_mean) / np.sqrt(
        np.maximum(predictive_variance, 1e-9)
    )
    residuals = data[[
        "user_id", "exercise_id", "day_index", "difficulty_level", "successful", "evaluated"
    ]].copy()
    residuals["predicted_probability"] = p_mean
    residuals["predicted_successful"] = evaluated * p_mean
    residuals["observed_rate"] = successful / evaluated
    residuals["predictive_variance"] = predictive_variance
    residuals["pearson_residual"] = residual
    metrics = {
        "joint_user_lpd": float(np.sum(logsumexp(user_ll, axis=0) - np.log(len(user_ll)))),
        "binary_log_loss": float(-np.sum(
            successful * np.log(np.clip(p_mean, 1e-9, 1))
            + failures * np.log(np.clip(1 - p_mean, 1e-9, 1))
        ) / evaluated.sum()),
        "brier_score": float(np.sum(
            successful * (1 - p_mean) ** 2 + failures * p_mean**2
        ) / evaluated.sum()),
        "pearson_residual_mean": float(residual.mean()),
        "pearson_residual_sd": float(residual.std(ddof=1)),
        "absolute_residual_p95": float(np.quantile(np.abs(residual), 0.95)),
        "kappa": float(kappa.mean()),
    }
    return metrics, residuals


def extract_student_slopes(
    result: FitResult,
    response_type: str,
    window: str,
    day_std: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_ids = [str(x) for x in result.model.backend.model.coords["user_id__factor_dim"]]
    theta_raw = values(result.trace, "1|user_id")
    lambda_raw = values(result.trace, "day_z|user_id")
    # Exact sample-wise centering provides an identified population mean and deviations.
    theta = theta_raw - theta_raw.mean(axis=1, keepdims=True)
    lambda_deviation = lambda_raw - lambda_raw.mean(axis=1, keepdims=True)
    global_slope = (
        values(result.trace, "day_z") + lambda_raw.mean(axis=1)
    ) / day_std
    total_slope = global_slope[:, None] + lambda_deviation / day_std
    deviation_per_day = lambda_deviation / day_std
    theta_summary = posterior_summary(theta)
    slope_summary = posterior_summary(total_slope)
    deviation_summary = posterior_summary(deviation_per_day)
    output = pd.DataFrame({
        "user_id": user_ids, "response_type": response_type, "window": window,
        "initial_effect": theta_summary["estimate"],
        "initial_effect_sd": theta_summary["posterior_sd"],
        "adjusted_trajectory_slope": slope_summary["estimate"],
        "slope_sd": slope_summary["posterior_sd"],
        "slope_hdi_3pct": slope_summary["hdi_3pct"],
        "slope_hdi_97pct": slope_summary["hdi_97pct"],
        "probability_slope_positive": (total_slope > 0).mean(axis=0),
        "random_slope_deviation": deviation_summary["estimate"],
        "probability_deviation_positive": (deviation_per_day > 0).mean(axis=0),
    })
    output["slope_class"] = "uncertain"
    output.loc[output.slope_hdi_3pct > 0, "slope_class"] = "positive"
    output.loc[output.slope_hdi_97pct < 0, "slope_class"] = "negative"
    sigma_draws = values(result.trace, "day_z|user_id_sigma") / day_std
    population = pd.DataFrame({
        "response_type": [response_type], "window": [window],
        **{
            f"sigma_lambda_{name}": [value[0]]
            for name, value in posterior_summary(sigma_draws[:, None]).items()
        },
        "global_slope_estimate": [global_slope.mean()],
        "global_slope_sd": [global_slope.std(ddof=1)],
        "probability_sigma_lambda_gt_0_001": [(sigma_draws > 0.001).mean()],
    })
    return output, population


def stability_summary(short: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    merged = short.merge(
        full, on=["user_id", "response_type"], suffixes=("_7_21", "_7_31")
    )
    rows = []
    for response_type, frame in merged.groupby("response_type", observed=True):
        left = frame.adjusted_trajectory_slope_7_21
        right = frame.adjusted_trajectory_slope_7_31
        rows.append({
            "response_type": response_type,
            "students_in_both_windows": len(frame),
            "pearson_correlation": float(left.corr(right)),
            "spearman_correlation": float(spearmanr(left, right).statistic),
            "sign_concordance": float((np.sign(left) == np.sign(right)).mean()),
            "classification_concordance": float(
                (frame.slope_class_7_21 == frame.slope_class_7_31).mean()
            ),
            "median_absolute_slope_difference": float(np.median(np.abs(left - right))),
        })
    return pd.DataFrame(rows)


def render_report(
    comparison: pd.DataFrame,
    population: pd.DataFrame,
    identifiability: pd.DataFrame,
    stability: pd.DataFrame,
    conditional: pd.DataFrame,
    metadata: dict,
) -> str:
    def markdown(frame: pd.DataFrame) -> list[str]:
        lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
        for row in frame.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(
                str(x) if not isinstance(x, (float, np.floating)) else f"{x:.4f}" for x in row
            ) + " |")
        return lines
    volume = conditional.loc[conditional.segment == "user_volume_quartile", [
        "response_type", "model", "segment_value", "evaluated", "calibration_error"
    ]]
    return "\n".join([
        "# Beta-binomial student random-slope experiment", "", "## Design", "",
        f"- User split: {metadata['train_users']} train / {metadata['test_users']} test.",
        "- Notes and chords fitted separately.",
        "- Random slopes are reported as adjusted future-performance change in log-odds per day.",
        "- `adjusted_trajectory_slope` includes the global slope plus the centered student deviation.",
        "", "## Held-out model comparison", "", *markdown(comparison), "",
        "## Population slope variation", "", *markdown(population), "",
        "## Individual slope identifiability", "", *markdown(identifiability), "",
        "## Temporal stability", "", *markdown(stability), "",
        "## Calibration error by held-out user volume", "", *markdown(volume), "",
        "## Interpretation", "",
        "Population slope variation is non-zero, but most individual intervals overlap zero. The random-slope model improves joint held-out LPD only modestly and does not improve element-level log loss or volume-conditioned mean calibration.",
        "",
        "Inference uses 20,000-iteration mean-field ADVI with 1,000 posterior draws. Mean-field ADVI can underestimate dependence and uncertainty, so the reported identifiable percentages may be optimistic and require a reduced-data NUTS check before target construction.",
        "",
        "These slopes describe change in platform-recorded, content-adjusted future performance. They are not yet labeled as learning.",
    ]) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/student_exercise_day.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/model_outputs/random_slope_v1"))
    parser.add_argument("--report", type=Path, default=Path("reports/random_slope_v1.md"))
    parser.add_argument("--vi-iterations", type=int, default=20000)
    parser.add_argument("--posterior-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    all_data = pd.read_parquet(args.input)
    full_window = prepare_data(all_data, 7, 31)
    train_users, test_users = split_users(full_window, 0.2, args.seed)
    comparisons, residual_frames = [], []
    slope_frames, population_frames = [], []

    for response_index, response_type in enumerate(("note", "chord")):
        response = full_window.loc[full_window.response_type == response_type].copy()
        train = response.loc[response.user_id.isin(train_users)].copy()
        test = response.loc[response.user_id.isin(test_users)].copy()
        train, test, _ = standardize(train, test)
        baseline_lpd = None
        for model_index, model_name in enumerate(("random_intercept", "random_slope")):
            print(f"Fitting held-out comparison: {response_type} {model_name}", flush=True)
            result = fit_model(
                train, model_name, args.vi_iterations, args.posterior_draws,
                args.seed + response_index * 10 + model_index,
            )
            metrics, residuals = evaluate(
                result, test, model_name, args.seed + 100 + response_index * 10 + model_index
            )
            if baseline_lpd is None:
                baseline_lpd = metrics["joint_user_lpd"]
            comparisons.append({
                "response_type": response_type, "model": model_name, **metrics,
                "delta_lpd_vs_random_intercept": metrics["joint_user_lpd"] - baseline_lpd,
                "elbo_final_loss": result.elbo_final,
            })
            residuals.insert(0, "response_type", response_type)
            residuals.insert(1, "model", model_name)
            residuals["family"] = model_name
            residual_frames.append(residuals)
            del result
            gc.collect()

        for window_name, day_end in (("7_21", 21), ("7_31", 31)):
            window = prepare_data(all_data, 7, day_end)
            window = window.loc[window.response_type == response_type].copy()
            window, _, scaling = standardize(window, window.iloc[:0].copy())
            print(f"Fitting slope extraction: {response_type} {window_name}", flush=True)
            result = fit_model(
                window, "random_slope", args.vi_iterations, args.posterior_draws,
                args.seed + 1000 + response_index * 10 + (0 if day_end == 21 else 1),
            )
            slopes, population = extract_student_slopes(
                result, response_type, window_name, scaling["day_index_std"]
            )
            slope_frames.append(slopes)
            population_frames.append(population)
            del result
            gc.collect()

    comparison = pd.DataFrame(comparisons)
    residuals = pd.concat(residual_frames, ignore_index=True)
    slopes = pd.concat(slope_frames, ignore_index=True)
    population = pd.concat(population_frames, ignore_index=True)
    full_slopes = slopes.loc[slopes.window == "7_31"]
    identifiability = (
        full_slopes.groupby(["response_type", "slope_class"], observed=True)
        .size().rename("students").reset_index()
    )
    totals = identifiability.groupby("response_type")["students"].transform("sum")
    identifiability["percentage"] = identifiability.students / totals
    stability = stability_summary(
        slopes.loc[slopes.window == "7_21"], full_slopes
    )
    conditional = conditional_calibration(residuals).rename(columns={"family": "model"})
    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
    residuals.to_parquet(args.output_dir / "heldout_residuals.parquet", index=False)
    conditional.to_csv(args.output_dir / "conditional_calibration.csv", index=False)
    slopes.to_parquet(args.output_dir / "student_adjusted_trajectory_slopes.parquet", index=False)
    population.to_csv(args.output_dir / "population_slope_variation.csv", index=False)
    identifiability.to_csv(args.output_dir / "slope_identifiability.csv", index=False)
    stability.to_csv(args.output_dir / "slope_stability.csv", index=False)
    metadata = {
        "train_users": len(train_users), "test_users": len(test_users),
        "vi_iterations": args.vi_iterations, "posterior_draws": args.posterior_draws,
        "seed": args.seed, "slope_unit": "log_odds_per_day",
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    report_comparison = comparison[[
        "response_type", "model", "joint_user_lpd", "delta_lpd_vs_random_intercept",
        "binary_log_loss", "brier_score", "pearson_residual_sd", "kappa",
    ]]
    args.report.write_text(
        render_report(
            report_comparison, population, identifiability, stability, conditional, metadata
        ),
        encoding="utf-8",
    )
    print(f"Wrote random-slope outputs to {args.output_dir}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
