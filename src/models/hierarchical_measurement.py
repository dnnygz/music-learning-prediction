#!/usr/bin/env python3
"""Fit incremental hierarchical binomial measurement models with Bambi/PyMC.

The note and chord processes are fitted separately. Model comparison uses a
user-level holdout; the final Model 4 is then refitted on all eligible rows to
export partially pooled latent effects and content-adjusted daily performance.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit, gammaln, logsumexp


RANDOM_SEED = 20260928
MODEL_TERMS = {
    "model_0": [],
    "model_1": ["difficulty_z"],
    "model_2": ["difficulty_z", "(1|exercise_id)"],
    "model_3": ["difficulty_z", "(1|exercise_id)", "(1|user_id)"],
    "model_4": ["difficulty_z", "day_z", "(1|exercise_id)", "(1|user_id)"],
}


@dataclass
class FitResult:
    model: bmb.Model
    trace: Any
    elbo_final: float


def formula_for(model_name: str) -> str:
    terms = MODEL_TERMS[model_name]
    rhs = "1" if not terms else "1 + " + " + ".join(terms)
    return f"proportion(successful, evaluated) ~ {rhs}"


def prepare_data(data: pd.DataFrame, day_start: int, day_end: int) -> pd.DataFrame:
    required = {
        "user_id", "exercise_id", "day_index", "difficulty_level",
        "response_type", "successful", "evaluated",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing modeling columns: {missing}")
    frame = data.loc[
        data["day_index"].between(day_start, day_end - 1)
        & (data["evaluated"] > 0)
    ].copy()
    invalid = (frame["successful"] < 0) | (frame["successful"] > frame["evaluated"])
    if invalid.any():
        raise ValueError(f"Found {int(invalid.sum())} invalid binomial rows")
    for column in ("user_id", "exercise_id", "response_type"):
        frame[column] = frame[column].astype(str)
    return frame.reset_index(drop=True)


def split_users(data: pd.DataFrame, test_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    users = np.array(sorted(data["user_id"].unique()))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(users)
    cut = int(round(len(users) * (1 - test_fraction)))
    return set(shuffled[:cut]), set(shuffled[cut:])


def standardize(train: pd.DataFrame, other: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    train, other = train.copy(), other.copy()
    values: dict[str, float] = {}
    for source, target in (("difficulty_level", "difficulty_z"), ("day_index", "day_z")):
        train_source = train[source].astype("float64")
        other_source = other[source].astype("float64")
        mean = float(train_source.mean())
        std = float(train_source.std(ddof=0))
        if not np.isfinite(std) or std == 0:
            std = 1.0
        train[target] = ((train_source - mean) / std).astype("float64")
        other[target] = ((other_source - mean) / std).astype("float64")
        values[f"{source}_mean"] = mean
        values[f"{source}_std"] = std
    return train, other, values


def priors_for(model_name: str) -> dict[str, bmb.Prior]:
    priors: dict[str, bmb.Prior] = {"Intercept": bmb.Prior("Normal", mu=0, sigma=1.5)}
    if model_name != "model_0":
        priors["difficulty_z"] = bmb.Prior("Normal", mu=0, sigma=1)
    if model_name == "model_4":
        priors["day_z"] = bmb.Prior("Normal", mu=0, sigma=0.5)
    group_prior = bmb.Prior(
        "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1)
    )
    if model_name in {"model_2", "model_3", "model_4"}:
        priors["1|exercise_id"] = group_prior
    if model_name in {"model_3", "model_4"}:
        priors["1|user_id"] = group_prior
    return priors


def fit_model(
    data: pd.DataFrame,
    model_name: str,
    vi_iterations: int,
    posterior_draws: int,
    seed: int,
) -> FitResult:
    model = bmb.Model(
        formula_for(model_name),
        data,
        family="binomial",
        priors=priors_for(model_name),
        categorical=["exercise_id", "user_id"],
    )
    approximation = model.fit(
        inference_method="vi",
        n=vi_iterations,
        method="advi",
        obj_optimizer=pm.adam(learning_rate=0.01),
        progressbar=False,
        random_seed=seed,
    )
    trace = approximation.sample(
        posterior_draws, return_inferencedata=False, random_seed=seed
    )
    return FitResult(model=model, trace=trace, elbo_final=float(approximation.hist[-1]))


def values(trace: Any, name: str) -> np.ndarray:
    return np.asarray(trace.get_values(name, combine=True))


def levels(model: bmb.Model, factor: str) -> list[str]:
    return [str(value) for value in model.backend.model.coords[f"{factor}__factor_dim"]]


def linear_predictor_draws(
    result: FitResult,
    data: pd.DataFrame,
    model_name: str,
    new_students: bool,
    seed: int,
    chunk_size: int = 5000,
):
    trace = result.trace
    intercept = values(trace, "Intercept")
    draws = len(intercept)
    difficulty = values(trace, "difficulty_z") if model_name != "model_0" else None
    day = values(trace, "day_z") if model_name == "model_4" else None

    exercise_effect = exercise_index = None
    if model_name in {"model_2", "model_3", "model_4"}:
        exercise_effect = values(trace, "1|exercise_id")
        exercise_index = {name: idx for idx, name in enumerate(levels(result.model, "exercise_id"))}

    student_effect = student_index = None
    sampled_new_student = None
    if model_name in {"model_3", "model_4"}:
        if new_students:
            sigma = values(trace, "1|user_id_sigma")
            unique_users = sorted(data["user_id"].unique())
            user_lookup = {name: idx for idx, name in enumerate(unique_users)}
            rng = np.random.default_rng(seed)
            sampled_new_student = rng.normal(size=(draws, len(unique_users))) * sigma[:, None]
            student_index = user_lookup
        else:
            student_effect = values(trace, "1|user_id")
            student_index = {name: idx for idx, name in enumerate(levels(result.model, "user_id"))}

    for start in range(0, len(data), chunk_size):
        frame = data.iloc[start : start + chunk_size]
        eta = np.broadcast_to(intercept[:, None], (draws, len(frame))).copy()
        if difficulty is not None:
            eta += difficulty[:, None] * frame["difficulty_z"].to_numpy()[None, :]
        if day is not None:
            eta += day[:, None] * frame["day_z"].to_numpy()[None, :]
        if exercise_effect is not None and exercise_index is not None:
            idx = np.array([exercise_index.get(value, -1) for value in frame["exercise_id"]])
            known = idx >= 0
            if known.any():
                eta[:, known] += exercise_effect[:, idx[known]]
        if student_index is not None:
            idx = np.array([student_index.get(value, -1) for value in frame["user_id"]])
            known = idx >= 0
            if known.any():
                effect = sampled_new_student if new_students else student_effect
                eta[:, known] += effect[:, idx[known]]
        yield start, expit(eta)


def binomial_logpmf(successful: np.ndarray, evaluated: np.ndarray, probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-9, 1 - 1e-9)
    combinatorial = gammaln(evaluated + 1) - gammaln(successful + 1) - gammaln(evaluated - successful + 1)
    return combinatorial + successful * np.log(probability) + (evaluated - successful) * np.log1p(-probability)


def evaluate(
    result: FitResult,
    data: pd.DataFrame,
    model_name: str,
    new_students: bool,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    successful = data["successful"].to_numpy(dtype=float)
    evaluated = data["evaluated"].to_numpy(dtype=float)
    p_mean = np.empty(len(data), dtype=float)
    lppd = np.empty(len(data), dtype=float)
    variance_log_likelihood = np.empty(len(data), dtype=float)
    user_names = sorted(data["user_id"].unique())
    user_lookup = {name: idx for idx, name in enumerate(user_names)}
    user_codes = data["user_id"].map(user_lookup).to_numpy()
    user_log_likelihood = None
    for start, probability in linear_predictor_draws(
        result, data, model_name, new_students=new_students, seed=seed
    ):
        stop = start + probability.shape[1]
        log_likelihood = binomial_logpmf(
            successful[None, start:stop], evaluated[None, start:stop], probability
        )
        p_mean[start:stop] = probability.mean(axis=0)
        lppd[start:stop] = logsumexp(log_likelihood, axis=0) - np.log(probability.shape[0])
        variance_log_likelihood[start:stop] = log_likelihood.var(axis=0, ddof=1)
        if user_log_likelihood is None:
            user_log_likelihood = np.zeros((probability.shape[0], len(user_names)))
        chunk_codes = user_codes[start:stop]
        for draw_index in range(probability.shape[0]):
            np.add.at(
                user_log_likelihood[draw_index], chunk_codes, log_likelihood[draw_index]
            )

    total_evaluated = evaluated.sum()
    failures = evaluated - successful
    binary_log_loss = -np.sum(
        successful * np.log(np.clip(p_mean, 1e-9, 1))
        + failures * np.log(np.clip(1 - p_mean, 1e-9, 1))
    ) / total_evaluated
    brier = np.sum(successful * (1 - p_mean) ** 2 + failures * p_mean**2) / total_evaluated
    pearson = (successful - evaluated * p_mean) / np.sqrt(
        np.maximum(evaluated * p_mean * (1 - p_mean), 1e-9)
    )
    residuals = data[["user_id", "exercise_id", "day_index", "successful", "evaluated"]].copy()
    residuals["predicted_probability"] = p_mean
    residuals["observed_rate"] = successful / evaluated
    residuals["pearson_residual"] = pearson

    bins = pd.cut(p_mean, bins=np.linspace(0, 1, 11), include_lowest=True, duplicates="drop")
    calibration_source = pd.DataFrame(
        {"bin": bins, "successful": successful, "evaluated": evaluated, "p_weighted": p_mean * evaluated}
    )
    calibration = calibration_source.groupby("bin", observed=True).agg(
        rows=("successful", "size"), successful=("successful", "sum"),
        evaluated=("evaluated", "sum"), p_weighted=("p_weighted", "sum"),
    ).reset_index()
    calibration["predicted_probability"] = calibration["p_weighted"] / calibration["evaluated"]
    calibration["observed_rate"] = calibration["successful"] / calibration["evaluated"]
    calibration["absolute_calibration_error"] = (
        calibration["predicted_probability"] - calibration["observed_rate"]
    ).abs()
    ece = float(
        np.average(calibration["absolute_calibration_error"], weights=calibration["evaluated"])
    )
    metrics = {
        # Group-level integration preserves the fact that one latent student
        # effect is shared by every held-out row from that student.
        "log_predictive_density": float(
            np.sum(logsumexp(user_log_likelihood, axis=0) - np.log(user_log_likelihood.shape[0]))
        ),
        "rowwise_lppd_diagnostic": float(lppd.sum()),
        "binary_log_loss": float(binary_log_loss),
        "brier_score": float(brier),
        "expected_calibration_error": ece,
        "pearson_residual_mean": float(np.mean(pearson)),
        "pearson_residual_sd": float(np.std(pearson, ddof=1)),
        "waic": float(-2 * (lppd.sum() - variance_log_likelihood.sum())),
        "p_waic": float(variance_log_likelihood.sum()),
        "rows": int(len(data)),
        "evaluated_elements": int(total_evaluated),
    }
    return metrics, residuals, calibration


def posterior_summary(draws: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "estimate": draws.mean(axis=0),
        "posterior_sd": draws.std(axis=0, ddof=1),
        "hdi_3pct": np.quantile(draws, 0.03, axis=0),
        "hdi_97pct": np.quantile(draws, 0.97, axis=0),
    }


def summarize_fixed(result: FitResult, model_name: str, response_type: str) -> pd.DataFrame:
    names = ["Intercept"]
    if model_name != "model_0":
        names.append("difficulty_z")
    if model_name == "model_4":
        names.append("day_z")
    rows = []
    for name in names:
        summary = posterior_summary(values(result.trace, name)[:, None])
        rows.append({"model": model_name, "response_type": response_type, "parameter": name,
                     **{key: float(value[0]) for key, value in summary.items()}})
    for factor in ("exercise_id", "user_id"):
        sigma_name = f"1|{factor}_sigma"
        if sigma_name in result.trace.varnames:
            summary = posterior_summary(values(result.trace, sigma_name)[:, None])
            rows.append({"model": model_name, "response_type": response_type,
                         "parameter": f"sigma_{factor.removesuffix('_id')}",
                         **{key: float(value[0]) for key, value in summary.items()}})
    return pd.DataFrame(rows)


def centered_effects(result: FitResult, factor: str, response_type: str) -> tuple[pd.DataFrame, np.ndarray]:
    raw = values(result.trace, f"1|{factor}")
    centered = raw - raw.mean(axis=1, keepdims=True)
    summary = posterior_summary(centered)
    output = pd.DataFrame({factor: levels(result.model, factor), "response_type": response_type})
    for name, value in summary.items():
        output[name] = value
    return output, centered


def temporal_effects(result: FitResult, response_type: str, day_mean: float, day_std: float) -> pd.DataFrame:
    coefficient = values(result.trace, "day_z")
    days = np.arange(7, 31)
    draws = coefficient[:, None] * ((days - day_mean) / day_std)[None, :]
    output = pd.DataFrame({"day_index": days, "response_type": response_type})
    for name, value in posterior_summary(draws).items():
        output[name] = value
    return output


def student_trajectories(
    result: FitResult,
    response_type: str,
    day_mean: float,
    day_std: float,
) -> pd.DataFrame:
    intercept = values(result.trace, "Intercept")
    exercise_raw = values(result.trace, "1|exercise_id")
    student_raw = values(result.trace, "1|user_id")
    # Reparameterize exported effects to exact sample-wise zero means.
    adjusted_intercept = intercept + exercise_raw.mean(axis=1) + student_raw.mean(axis=1)
    student = student_raw - student_raw.mean(axis=1, keepdims=True)
    day_coefficient = values(result.trace, "day_z")
    days = np.arange(7, 31)
    day_z = (days - day_mean) / day_std
    probability = expit(
        adjusted_intercept[:, None, None]
        + student[:, :, None]
        + day_coefficient[:, None, None] * day_z[None, None, :]
    )
    centered_days = days - days.mean()
    slope = np.sum(probability * centered_days[None, None, :], axis=2) / np.sum(centered_days**2)
    users = levels(result.model, "user_id")
    summary = posterior_summary(probability)
    slope_summary = posterior_summary(slope)
    output = pd.MultiIndex.from_product(
        [users, days], names=["user_id", "day_index"]
    ).to_frame(index=False)
    output["response_type"] = response_type
    for name, values_array in summary.items():
        output[f"estimated_performance_{name}"] = values_array.reshape(-1)
    for name, values_array in slope_summary.items():
        output[f"trajectory_slope_{name}"] = np.repeat(values_array, len(days))
    output["trajectory_uncertainty"] = output["trajectory_slope_posterior_sd"]
    return output


def render_report(comparison: pd.DataFrame, metadata: dict[str, Any]) -> str:
    display_columns = [
        "response_type", "model", "test_log_predictive_density",
        "delta_test_lpd_vs_previous", "test_binary_log_loss",
        "test_brier_score", "test_expected_calibration_error", "train_waic",
    ]
    display = comparison[display_columns].copy()
    headers = list(display.columns)
    markdown_rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("" if np.isnan(value) else f"{value:.4f}")
            else:
                cells.append(str(value))
        markdown_rows.append("| " + " | ".join(cells) + " |")
    conclusions = []
    diagnostics = []
    for response_type, frame in comparison.groupby("response_type", sort=False):
        frame = frame.set_index("model")
        conclusions.append(f"### {response_type.capitalize()}")
        for model_name in ("model_1", "model_2", "model_3", "model_4"):
            delta = frame.loc[model_name, "delta_test_lpd_vs_previous"]
            direction = "improved" if delta > 0 else "worsened"
            conclusions.append(
                f"- `{model_name}` {direction} held-out joint user LPD by {abs(delta):,.1f}."
            )
        best_log_loss = frame["test_binary_log_loss"].idxmin()
        model_4 = frame.loc["model_4"]
        diagnostics.append(
            f"- {response_type.capitalize()}: lowest element-level log loss is "
            f"`{best_log_loss}` ({frame.loc[best_log_loss, 'test_binary_log_loss']:.4f}); "
            f"Model 4 residual SD is {model_4['test_pearson_residual_sd']:.2f} and "
            f"calibration error is {model_4['test_expected_calibration_error']:.4f}."
        )
    lines = [
        "# Incremental hierarchical measurement model", "",
        "## Scope", "",
        f"- Measurement window: `[{metadata['day_start']}, {metadata['day_end']})`.",
        f"- User-level split: {metadata['train_users']} train / {metadata['test_users']} test.",
        "- Notes and chords were fitted as separate binomial model sequences.",
        "- Inference: mean-field ADVI through Bambi/PyMC; uncertainty is approximate.",
        "- AIC/BIC are not reported because these are Bayesian hierarchical fits; WAIC and held-out log predictive density are used instead.",
        "", "## Held-out comparison", "",
        *markdown_rows, "",
        "The held-out log predictive density integrates all rows from a user jointly, preserving the shared latent student effect. Element-level log loss, Brier score, and calibration are complementary and can disagree with this distributional criterion.",
        "", "## Incremental contribution", "", *conclusions, "",
        "## Residual diagnostics", "", *diagnostics, "",
        "Residual standard deviations far above one indicate substantial extra-binomial dispersion remains. This supports evaluating a beta-binomial sensitivity model after the binomial baseline. WAIC from mean-field ADVI is retained as a diagnostic but is not used for model selection because variational posterior variance can be underestimated or distorted.",
        "",
        "## Interpretation constraints", "",
        "Model 4 has a student random intercept and a global linear time effect. It can export adjusted daily performance, but it does not yet estimate a distinct latent learning-rate parameter for each student. Probability-scale slopes vary with baseline performance through the logistic link; a later random-slope model is required before interpreting slope differences as individual learning trajectories.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/student_exercise_day.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/model_outputs/hierarchical_v1"))
    parser.add_argument("--report", type=Path, default=Path("reports/hierarchical_measurement_v1.md"))
    parser.add_argument("--day-start", type=int, default=7)
    parser.add_argument("--day-end", type=int, default=31)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--vi-iterations", type=int, default=10000)
    parser.add_argument("--posterior-draws", type=int, default=400)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.test_fraction < 1:
        raise ValueError("test-fraction must be between zero and one")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    data = prepare_data(pd.read_parquet(args.input), args.day_start, args.day_end)
    train_users, test_users = split_users(data, args.test_fraction, args.seed)
    metric_rows, residual_frames, calibration_frames, fixed_frames = [], [], [], []
    exercise_frames, student_frames, temporal_frames, trajectory_frames = [], [], [], []

    for response_type in ("note", "chord"):
        response = data.loc[data["response_type"] == response_type].copy()
        train = response.loc[response["user_id"].isin(train_users)].copy()
        test = response.loc[response["user_id"].isin(test_users)].copy()
        train, test, scaling = standardize(train, test)
        previous_test_lpd = None
        for index, model_name in enumerate(MODEL_TERMS):
            print(f"Fitting {response_type} {model_name} ({len(train):,} train rows)", flush=True)
            result = fit_model(
                train, model_name, args.vi_iterations, args.posterior_draws, args.seed + index
            )
            train_metrics, _, _ = evaluate(
                result, train, model_name, new_students=False, seed=args.seed + index
            )
            test_metrics, residuals, calibration = evaluate(
                result, test, model_name, new_students=True, seed=args.seed + 100 + index
            )
            improvement = np.nan if previous_test_lpd is None else (
                test_metrics["log_predictive_density"] - previous_test_lpd
            )
            previous_test_lpd = test_metrics["log_predictive_density"]
            metric_rows.append({
                "response_type": response_type, "model": model_name,
                "formula": formula_for(model_name), "inference": "meanfield_advi",
                "elbo_final_loss": result.elbo_final,
                "train_log_predictive_density": train_metrics["log_predictive_density"],
                "train_waic": train_metrics["waic"],
                "test_log_predictive_density": test_metrics["log_predictive_density"],
                "delta_test_lpd_vs_previous": improvement,
                "test_binary_log_loss": test_metrics["binary_log_loss"],
                "test_brier_score": test_metrics["brier_score"],
                "test_expected_calibration_error": test_metrics["expected_calibration_error"],
                "test_pearson_residual_mean": test_metrics["pearson_residual_mean"],
                "test_pearson_residual_sd": test_metrics["pearson_residual_sd"],
                "aic": np.nan, "bic": np.nan,
            })
            residuals.insert(0, "response_type", response_type)
            residuals.insert(1, "model", model_name)
            calibration.insert(0, "response_type", response_type)
            calibration.insert(1, "model", model_name)
            residual_frames.append(residuals)
            calibration_frames.append(calibration)
            fixed_frames.append(summarize_fixed(result, model_name, response_type))
            del result
            gc.collect()

        # Refit Model 4 on all future measurement rows for latent-effect exports.
        response, _, full_scaling = standardize(response, response.iloc[:0].copy())
        print(f"Refitting {response_type} model_4 on all {len(response):,} rows", flush=True)
        final = fit_model(
            response, "model_4", args.vi_iterations, args.posterior_draws, args.seed + 1000
        )
        exercise, _ = centered_effects(final, "exercise_id", response_type)
        student, _ = centered_effects(final, "user_id", response_type)
        temporal = temporal_effects(
            final, response_type, full_scaling["day_index_mean"], full_scaling["day_index_std"]
        )
        trajectories = student_trajectories(
            final, response_type, full_scaling["day_index_mean"], full_scaling["day_index_std"]
        )
        exercise.to_parquet(args.output_dir / f"exercise_effects_{response_type}.parquet", index=False)
        student.to_parquet(args.output_dir / f"student_effects_{response_type}.parquet", index=False)
        temporal.to_parquet(args.output_dir / f"temporal_effects_{response_type}.parquet", index=False)
        trajectories.to_parquet(args.output_dir / f"student_trajectory_{response_type}.parquet", index=False)
        exercise_frames.append(exercise)
        student_frames.append(student)
        temporal_frames.append(temporal)
        trajectory_frames.append(trajectories)
        del final
        gc.collect()

    comparison = pd.DataFrame(metric_rows)
    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
    pd.concat(fixed_frames, ignore_index=True).to_csv(
        args.output_dir / "fixed_parameter_summary.csv", index=False
    )
    pd.concat(residual_frames, ignore_index=True).to_parquet(
        args.output_dir / "heldout_residuals.parquet", index=False
    )
    pd.concat(calibration_frames, ignore_index=True).to_csv(
        args.output_dir / "heldout_calibration.csv", index=False
    )
    pd.concat(exercise_frames, ignore_index=True).to_parquet(
        args.output_dir / "exercise_effects.parquet", index=False
    )
    pd.concat(student_frames, ignore_index=True).to_parquet(
        args.output_dir / "student_effects.parquet", index=False
    )
    pd.concat(temporal_frames, ignore_index=True).to_parquet(
        args.output_dir / "temporal_effects.parquet", index=False
    )
    pd.concat(trajectory_frames, ignore_index=True).to_parquet(
        args.output_dir / "student_trajectory.parquet", index=False
    )
    metadata = {
        "input": str(args.input), "day_start": args.day_start, "day_end": args.day_end,
        "train_users": len(train_users), "test_users": len(test_users),
        "split_seed": args.seed, "vi_iterations": args.vi_iterations,
        "posterior_draws": args.posterior_draws,
        "aic_bic": "not_applicable_for_bayesian_hierarchical_models",
        "trajectory_limitation": "global_time_slope_only_no_student_random_slope",
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.report.write_text(render_report(comparison, metadata), encoding="utf-8")
    print(f"Wrote model outputs to {args.output_dir}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
