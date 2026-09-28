#!/usr/bin/env python3
"""Compare linear and small-spline beta-binomial mean structures."""

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

from src.models.measurement.dispersion_comparison import conditional_calibration, logpmf
from src.models.measurement.hierarchical_measurement import (
    RANDOM_SEED, FitResult, posterior_summary, prepare_data, split_users,
    standardize, values,
)


FORMULAS = {
    "linear": (
        "proportion(successful, evaluated) ~ 1 + difficulty_z + day_z "
        "+ (1|exercise_id) + (1|user_id)"
    ),
    "spline": (
        "proportion(successful, evaluated) ~ 1 + bs(difficulty_z, df=5) "
        "+ bs(day_z, df=5) + (1|exercise_id) + (1|user_id)"
    ),
}


def priors(model_name: str) -> dict[str, bmb.Prior]:
    output = {
        "Intercept": bmb.Prior("Normal", mu=0, sigma=1.5),
        "1|exercise_id": bmb.Prior(
            "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1)
        ),
        "1|user_id": bmb.Prior(
            "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1)
        ),
        "kappa": bmb.Prior("HalfNormal", sigma=50),
    }
    if model_name == "linear":
        output["difficulty_z"] = bmb.Prior("Normal", mu=0, sigma=1)
        output["day_z"] = bmb.Prior("Normal", mu=0, sigma=0.5)
    else:
        output["bs(difficulty_z, df=5)"] = bmb.Prior("Normal", mu=0, sigma=1)
        output["bs(day_z, df=5)"] = bmb.Prior("Normal", mu=0, sigma=0.5)
    return output


def fit(data: pd.DataFrame, model_name: str, iterations: int, draws: int, seed: int) -> FitResult:
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


def common_coefficients(result: FitResult, model_name: str) -> np.ndarray:
    arrays = [values(result.trace, "Intercept")[:, None]]
    if model_name == "linear":
        arrays.extend([
            values(result.trace, "difficulty_z")[:, None],
            values(result.trace, "day_z")[:, None],
        ])
    else:
        arrays.extend([
            values(result.trace, "bs(difficulty_z, df=5)"),
            values(result.trace, "bs(day_z, df=5)"),
        ])
    return np.concatenate(arrays, axis=1)


def probability_draws(
    result: FitResult, data: pd.DataFrame, model_name: str, seed: int, chunk_size: int = 5000
):
    coefficients = common_coefficients(result, model_name)
    exercise_effect = values(result.trace, "1|exercise_id")
    exercise_levels = [str(x) for x in result.model.backend.model.coords["exercise_id__factor_dim"]]
    exercise_lookup = {name: index for index, name in enumerate(exercise_levels)}
    sigma_student = values(result.trace, "1|user_id_sigma")
    users = sorted(data["user_id"].unique())
    user_lookup = {name: index for index, name in enumerate(users)}
    rng = np.random.default_rng(seed)
    new_student = rng.normal(size=(len(coefficients), len(users))) * sigma_student[:, None]
    # Formulae retains fitted spline knots and evaluates new data consistently.
    common_design = result.model.components["mu"].design.common
    for start in range(0, len(data), chunk_size):
        frame = data.iloc[start : start + chunk_size]
        design = np.asarray(common_design.evaluate_new_data(frame))
        eta = coefficients @ design.T
        exercise_idx = np.array([exercise_lookup.get(x, -1) for x in frame["exercise_id"]])
        known = exercise_idx >= 0
        if known.any():
            eta[:, known] += exercise_effect[:, exercise_idx[known]]
        user_idx = np.array([user_lookup[x] for x in frame["user_id"]])
        eta += new_student[:, user_idx]
        yield start, expit(eta)


def evaluate(result: FitResult, data: pd.DataFrame, model_name: str, seed: int):
    successful = data["successful"].to_numpy(float)
    evaluated = data["evaluated"].to_numpy(float)
    kappa = values(result.trace, "kappa")
    p_mean = np.empty(len(data))
    variance = np.empty(len(data))
    users = sorted(data["user_id"].unique())
    user_lookup = {name: index for index, name in enumerate(users)}
    user_codes = data["user_id"].map(user_lookup).to_numpy()
    user_ll = None
    for start, probability in probability_draws(result, data, model_name, seed):
        stop = start + probability.shape[1]
        n = evaluated[None, start:stop]
        y = successful[None, start:stop]
        ll = logpmf("beta_binomial", y, n, probability, kappa)
        if user_ll is None:
            user_ll = np.zeros((len(probability), len(users)))
        for draw_index in range(len(probability)):
            np.add.at(user_ll[draw_index], user_codes[start:stop], ll[draw_index])
        mean_count = n * probability
        conditional_variance = (
            n * probability * (1 - probability) * (n + kappa[:, None])
            / (kappa[:, None] + 1)
        )
        p_mean[start:stop] = probability.mean(axis=0)
        variance[start:stop] = conditional_variance.mean(axis=0) + mean_count.var(axis=0, ddof=1)
    failures = evaluated - successful
    residual = (successful - evaluated * p_mean) / np.sqrt(np.maximum(variance, 1e-9))
    residuals = data[[
        "user_id", "exercise_id", "day_index", "difficulty_level", "successful", "evaluated"
    ]].copy()
    residuals["predicted_probability"] = p_mean
    residuals["predicted_successful"] = evaluated * p_mean
    residuals["observed_rate"] = successful / evaluated
    residuals["predictive_variance"] = variance
    residuals["pearson_residual"] = residual
    calibration_error = np.sum(np.abs(residuals["predicted_successful"] - successful)) / evaluated.sum()
    metrics = {
        "joint_user_lpd": float(np.sum(logsumexp(user_ll, axis=0) - np.log(len(user_ll)))),
        "binary_log_loss": float(-np.sum(
            successful * np.log(np.clip(p_mean, 1e-9, 1))
            + failures * np.log(np.clip(1 - p_mean, 1e-9, 1))
        ) / evaluated.sum()),
        "brier_score": float(np.sum(
            successful * (1 - p_mean) ** 2 + failures * p_mean**2
        ) / evaluated.sum()),
        "weighted_absolute_row_error": float(calibration_error),
        "pearson_residual_sd": float(residual.std(ddof=1)),
        "kappa": float(kappa.mean()),
    }
    return metrics, residuals


def render_report(comparison: pd.DataFrame, conditional: pd.DataFrame, metadata: dict) -> str:
    columns = [
        "response_type", "model", "joint_user_lpd", "delta_lpd_vs_linear",
        "binary_log_loss", "brier_score", "weighted_absolute_row_error",
        "pearson_residual_sd", "kappa",
    ]
    table = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in comparison[columns].itertuples(index=False, name=None):
        table.append("| " + " | ".join(
            str(x) if not isinstance(x, (float, np.floating)) else f"{x:.4f}" for x in row
        ) + " |")
    findings = []
    for response_type, frame in comparison.groupby("response_type", sort=False):
        frame = frame.set_index("model")
        linear, spline = frame.loc["linear"], frame.loc["spline"]
        difficulty = conditional[
            (conditional.response_type == response_type)
            & (conditional.model == "spline")
            & (conditional.segment == "difficulty")
            & (conditional.evaluated >= 5000)
        ]
        max_error = difficulty.loc[difficulty.calibration_error.abs().idxmax()]
        findings.append(
            f"- {response_type.capitalize()}: spline changes LPD by "
            f"{spline.joint_user_lpd - linear.joint_user_lpd:,.1f} and log loss by "
            f"{spline.binary_log_loss - linear.binary_log_loss:+.4f}. Its largest supported "
            f"difficulty calibration error is {max_error.calibration_error:+.3f} at level "
            f"{max_error.segment_value}."
        )
    return "\n".join([
        "# Nonlinear beta-binomial bridge model", "", "## Design", "",
        f"- User split: {metadata['train_users']} train / {metadata['test_users']} test.",
        "- Linear baseline: standardized difficulty and day.",
        "- Bridge: cubic B-splines with 5 degrees of freedom for difficulty and day.",
        "- The likelihood and hierarchical student/exercise structure are otherwise identical.",
        "", "## Held-out comparison", "", *table, "", "## Findings", "",
        *findings, "", "The bridge is accepted only if it improves mean calibration without sacrificing the beta-binomial residual behavior. No additional spline sizes are searched.",
    ]) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/student_exercise_day.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/model_outputs/nonlinear_bridge_v1"))
    parser.add_argument("--report", type=Path, default=Path("reports/measurement/nonlinear_bridge_v1.md"))
    parser.add_argument("--vi-iterations", type=int, default=20000)
    parser.add_argument("--posterior-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    data = prepare_data(pd.read_parquet(args.input), 7, 31)
    train_users, test_users = split_users(data, 0.2, args.seed)
    rows, residual_frames = [], []
    for response_index, response_type in enumerate(("note", "chord")):
        response = data.loc[data.response_type == response_type].copy()
        train = response.loc[response.user_id.isin(train_users)].copy()
        test = response.loc[response.user_id.isin(test_users)].copy()
        train, test, _ = standardize(train, test)
        baseline_lpd = None
        for model_index, model_name in enumerate(("linear", "spline")):
            print(f"Fitting {response_type} {model_name}", flush=True)
            result = fit(
                train, model_name, args.vi_iterations, args.posterior_draws,
                args.seed + response_index * 10 + model_index,
            )
            metrics, residuals = evaluate(
                result, test, model_name, args.seed + 100 + response_index * 10 + model_index
            )
            if baseline_lpd is None:
                baseline_lpd = metrics["joint_user_lpd"]
            metrics.update({
                "response_type": response_type, "model": model_name,
                "delta_lpd_vs_linear": metrics["joint_user_lpd"] - baseline_lpd,
                "elbo_final_loss": result.elbo_final,
            })
            rows.append(metrics)
            residuals.insert(0, "response_type", response_type)
            residuals.insert(1, "model", model_name)
            # Reuse calibration utility, which expects a family column.
            residuals["family"] = model_name
            residual_frames.append(residuals)
            del result
            gc.collect()
    comparison = pd.DataFrame(rows)
    residuals = pd.concat(residual_frames, ignore_index=True)
    conditional = conditional_calibration(residuals).rename(columns={"family": "model"})
    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
    residuals.to_parquet(args.output_dir / "heldout_residuals.parquet", index=False)
    conditional.to_csv(args.output_dir / "conditional_calibration.csv", index=False)
    metadata = {
        "train_users": len(train_users), "test_users": len(test_users),
        "vi_iterations": args.vi_iterations, "posterior_draws": args.posterior_draws,
        "spline_df_difficulty": 5, "spline_df_day": 5, "seed": args.seed,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.report.write_text(render_report(comparison, conditional, metadata), encoding="utf-8")
    print(f"Wrote bridge comparison to {args.output_dir}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
