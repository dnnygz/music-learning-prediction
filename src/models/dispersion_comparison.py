#!/usr/bin/env python3
"""Compare hierarchical binomial and beta-binomial observation models."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import betaln, gammaln, logsumexp

from src.models.hierarchical_measurement import (
    RANDOM_SEED,
    FitResult,
    formula_for,
    linear_predictor_draws,
    posterior_summary,
    prepare_data,
    priors_for,
    split_users,
    standardize,
    values,
)


def fit_family(
    data: pd.DataFrame,
    family: str,
    vi_iterations: int,
    posterior_draws: int,
    seed: int,
) -> FitResult:
    priors = priors_for("model_4")
    if family == "beta_binomial":
        # Concentration is positive. This weak prior permits both substantial
        # overdispersion and convergence toward the binomial limit.
        priors["kappa"] = bmb.Prior("HalfNormal", sigma=50)
    model = bmb.Model(
        formula_for("model_4"), data, family=family, priors=priors,
        categorical=["exercise_id", "user_id"],
    )
    approximation = model.fit(
        inference_method="vi", n=vi_iterations, method="advi",
        obj_optimizer=pm.adam(learning_rate=0.01), progressbar=False,
        random_seed=seed,
    )
    trace = approximation.sample(
        posterior_draws, return_inferencedata=False, random_seed=seed
    )
    return FitResult(model=model, trace=trace, elbo_final=float(approximation.hist[-1]))


def logpmf(
    family: str,
    successful: np.ndarray,
    evaluated: np.ndarray,
    probability: np.ndarray,
    kappa: np.ndarray | None,
) -> np.ndarray:
    probability = np.clip(probability, 1e-9, 1 - 1e-9)
    combinatorial = (
        gammaln(evaluated + 1)
        - gammaln(successful + 1)
        - gammaln(evaluated - successful + 1)
    )
    if family == "binomial":
        return (
            combinatorial
            + successful * np.log(probability)
            + (evaluated - successful) * np.log1p(-probability)
        )
    alpha = probability * kappa[:, None]
    beta = (1 - probability) * kappa[:, None]
    return (
        combinatorial
        + betaln(successful + alpha, evaluated - successful + beta)
        - betaln(alpha, beta)
    )


def evaluate_family(
    result: FitResult,
    data: pd.DataFrame,
    family: str,
    new_students: bool,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    successful = data["successful"].to_numpy(dtype=float)
    evaluated = data["evaluated"].to_numpy(dtype=float)
    user_names = sorted(data["user_id"].unique())
    user_lookup = {name: idx for idx, name in enumerate(user_names)}
    user_codes = data["user_id"].map(user_lookup).to_numpy()
    kappa = values(result.trace, "kappa") if family == "beta_binomial" else None
    p_mean = np.empty(len(data))
    predictive_variance = np.empty(len(data))
    user_log_likelihood = None

    for start, probability in linear_predictor_draws(
        result, data, "model_4", new_students=new_students, seed=seed
    ):
        stop = start + probability.shape[1]
        y = successful[None, start:stop]
        n = evaluated[None, start:stop]
        ll = logpmf(family, y, n, probability, kappa)
        if user_log_likelihood is None:
            user_log_likelihood = np.zeros((probability.shape[0], len(user_names)))
        codes = user_codes[start:stop]
        for draw_index in range(probability.shape[0]):
            np.add.at(user_log_likelihood[draw_index], codes, ll[draw_index])

        conditional_mean = n * probability
        if family == "binomial":
            conditional_variance = n * probability * (1 - probability)
        else:
            conditional_variance = (
                n * probability * (1 - probability) * (n + kappa[:, None])
                / (kappa[:, None] + 1)
            )
        mean_count = conditional_mean.mean(axis=0)
        p_mean[start:stop] = probability.mean(axis=0)
        predictive_variance[start:stop] = (
            conditional_variance.mean(axis=0)
            + conditional_mean.var(axis=0, ddof=1)
        )

    failures = evaluated - successful
    total = evaluated.sum()
    binary_log_loss = -np.sum(
        successful * np.log(np.clip(p_mean, 1e-9, 1))
        + failures * np.log(np.clip(1 - p_mean, 1e-9, 1))
    ) / total
    brier = np.sum(successful * (1 - p_mean) ** 2 + failures * p_mean**2) / total
    residual = (successful - evaluated * p_mean) / np.sqrt(
        np.maximum(predictive_variance, 1e-9)
    )
    residuals = data[
        ["user_id", "exercise_id", "day_index", "difficulty_level", "successful", "evaluated"]
    ].copy()
    residuals["predicted_probability"] = p_mean
    residuals["predicted_successful"] = evaluated * p_mean
    residuals["observed_rate"] = successful / evaluated
    residuals["predictive_variance"] = predictive_variance
    residuals["pearson_residual"] = residual

    bins = pd.cut(p_mean, np.linspace(0, 1, 11), include_lowest=True)
    source = pd.DataFrame({
        "bin": bins, "successful": successful, "evaluated": evaluated,
        "p_weighted": p_mean * evaluated,
    })
    calibration = source.groupby("bin", observed=True).agg(
        rows=("successful", "size"), successful=("successful", "sum"),
        evaluated=("evaluated", "sum"), p_weighted=("p_weighted", "sum"),
    ).reset_index()
    calibration["predicted_probability"] = calibration["p_weighted"] / calibration["evaluated"]
    calibration["observed_rate"] = calibration["successful"] / calibration["evaluated"]
    calibration["absolute_calibration_error"] = (
        calibration["predicted_probability"] - calibration["observed_rate"]
    ).abs()
    ece = float(np.average(
        calibration["absolute_calibration_error"], weights=calibration["evaluated"]
    ))
    joint_lpd = float(np.sum(
        logsumexp(user_log_likelihood, axis=0) - np.log(user_log_likelihood.shape[0])
    ))
    metrics = {
        "joint_user_log_predictive_density": joint_lpd,
        "binary_log_loss": float(binary_log_loss),
        "brier_score": float(brier),
        "expected_calibration_error": ece,
        "pearson_residual_mean": float(residual.mean()),
        "pearson_residual_sd": float(residual.std(ddof=1)),
        "absolute_residual_p95": float(np.quantile(np.abs(residual), 0.95)),
    }
    return metrics, residuals, calibration


def dispersion_summary(result: FitResult, median_n: float) -> dict[str, float]:
    kappa = values(result.trace, "kappa")
    rho = 1 / (kappa + 1)
    vif = 1 + (median_n - 1) * rho
    summary = {}
    for prefix, draws in (("kappa", kappa), ("rho", rho), ("vif_at_median_n", vif)):
        values_summary = posterior_summary(draws[:, None])
        for statistic, value in values_summary.items():
            summary[f"{prefix}_{statistic}"] = float(value[0])
    return summary


def parameter_summary(result: FitResult, response_type: str, family: str) -> pd.DataFrame:
    names = [
        "Intercept", "difficulty_z", "day_z", "1|exercise_id_sigma", "1|user_id_sigma"
    ]
    if family == "beta_binomial":
        names.append("kappa")
    rows = []
    for name in names:
        summary = posterior_summary(values(result.trace, name)[:, None])
        rows.append({
            "response_type": response_type, "family": family, "parameter": name,
            **{key: float(value[0]) for key, value in summary.items()},
        })
    return pd.DataFrame(rows)


def conditional_calibration(residuals: pd.DataFrame) -> pd.DataFrame:
    frame = residuals.copy()
    user_volume = frame.groupby(
        ["response_type", "family", "user_id"], observed=True
    )["evaluated"].sum().rename("user_evaluated_volume")
    frame = frame.join(user_volume, on=["response_type", "family", "user_id"])
    frames = []
    for (response_type, family), part in frame.groupby(
        ["response_type", "family"], observed=True
    ):
        part = part.copy()
        try:
            part["volume_segment"] = pd.qcut(
                part["user_evaluated_volume"], 4,
                labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop",
            ).astype(str)
        except ValueError:
            part["volume_segment"] = "all"
        for segment_name, segment_column in (
            ("day", "day_index"),
            ("difficulty", "difficulty_level"),
            ("user_volume_quartile", "volume_segment"),
        ):
            grouped = part.groupby(segment_column, observed=True).agg(
                rows=("successful", "size"), successful=("successful", "sum"),
                evaluated=("evaluated", "sum"),
                predicted_successful=("predicted_successful", "sum"),
            ).reset_index().rename(columns={segment_column: "segment_value"})
            grouped["response_type"] = response_type
            grouped["family"] = family
            grouped["segment"] = segment_name
            grouped["predicted_probability"] = grouped["predicted_successful"] / grouped["evaluated"]
            grouped["observed_rate"] = grouped["successful"] / grouped["evaluated"]
            grouped["calibration_error"] = grouped["predicted_probability"] - grouped["observed_rate"]
            frames.append(grouped)
    return pd.concat(frames, ignore_index=True)


def render_report(comparison: pd.DataFrame, dispersion: pd.DataFrame, metadata: dict[str, Any]) -> str:
    columns = [
        "response_type", "family", "joint_user_log_predictive_density",
        "delta_lpd_vs_binomial", "binary_log_loss", "brier_score",
        "expected_calibration_error", "pearson_residual_sd",
        "absolute_residual_p95",
    ]
    table = comparison[columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(
            str(value) if not isinstance(value, (float, np.floating))
            else ("" if np.isnan(value) else f"{value:.4f}") for value in row
        ) + " |")
    findings = []
    for response_type, frame in comparison.groupby("response_type", sort=False):
        frame = frame.set_index("family")
        beta = frame.loc["beta_binomial"]
        d = dispersion.set_index("response_type").loc[response_type]
        findings.append(
            f"- {response_type.capitalize()}: beta-binomial changes held-out joint LPD by "
            f"{beta['delta_lpd_vs_binomial']:,.1f}; residual SD changes from "
            f"{frame.loc['binomial', 'pearson_residual_sd']:.2f} to "
            f"{beta['pearson_residual_sd']:.2f}. Estimated kappa is "
            f"{d['kappa_estimate']:.2f} and rho is {d['rho_estimate']:.4f}."
        )
    return "\n".join([
        "# Binomial versus beta-binomial dispersion diagnostic", "",
        "## Design", "",
        f"- Measurement window: `[{metadata['day_start']}, {metadata['day_end']})`.",
        f"- User holdout: {metadata['train_users']} train / {metadata['test_users']} test.",
        "- Both families use Model 4 predictors and identical zero-centered hierarchical priors.",
        "- Notes and chords are fitted separately.",
        "- Held-out likelihood integrates all rows from each student jointly.",
        "", "## Comparison", "", *lines, "", "## Findings", "", *findings, "",
        "Beta-binomial Pearson residuals use its posterior predictive variance, including parameter uncertainty. A residual SD closer to one indicates that the observation family accounts for more of the remaining count variability.",
        "", "## Limitation", "",
        "Inference uses mean-field ADVI. This is an efficient screening comparison; final uncertainty and the dispersion parameter should be checked with stronger variational diagnostics or NUTS before freezing the target.",
    ]) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/student_exercise_day.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/model_outputs/dispersion_v1"))
    parser.add_argument("--report", type=Path, default=Path("reports/dispersion_comparison_v1.md"))
    parser.add_argument("--day-start", type=int, default=7)
    parser.add_argument("--day-end", type=int, default=31)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--vi-iterations", type=int, default=3000)
    parser.add_argument("--posterior-draws", type=int, default=300)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    data = prepare_data(pd.read_parquet(args.input), args.day_start, args.day_end)
    train_users, test_users = split_users(data, args.test_fraction, args.seed)
    metric_rows, residual_frames, calibration_frames, dispersion_rows = [], [], [], []
    parameter_frames = []

    for response_type in ("note", "chord"):
        response = data.loc[data["response_type"] == response_type].copy()
        train = response.loc[response["user_id"].isin(train_users)].copy()
        test = response.loc[response["user_id"].isin(test_users)].copy()
        train, test, _ = standardize(train, test)
        family_metrics: dict[str, dict[str, float]] = {}
        for family_index, family in enumerate(("binomial", "beta_binomial")):
            print(f"Fitting {response_type} {family}", flush=True)
            result = fit_family(
                train, family, args.vi_iterations, args.posterior_draws,
                args.seed + family_index,
            )
            metrics, residuals, calibration = evaluate_family(
                result, test, family, new_students=True,
                seed=args.seed + 100 + family_index,
            )
            metrics.update({
                "response_type": response_type, "family": family,
                "elbo_final_loss": result.elbo_final,
            })
            family_metrics[family] = metrics
            residuals.insert(0, "response_type", response_type)
            residuals.insert(1, "family", family)
            calibration.insert(0, "response_type", response_type)
            calibration.insert(1, "family", family)
            residual_frames.append(residuals)
            calibration_frames.append(calibration)
            parameter_frames.append(parameter_summary(result, response_type, family))
            if family == "beta_binomial":
                dispersion_rows.append({
                    "response_type": response_type,
                    "median_evaluated_per_row": float(train["evaluated"].median()),
                    **dispersion_summary(result, float(train["evaluated"].median())),
                })
            del result
            gc.collect()
        baseline = family_metrics["binomial"]["joint_user_log_predictive_density"]
        for family in ("binomial", "beta_binomial"):
            row = family_metrics[family]
            row["delta_lpd_vs_binomial"] = (
                row["joint_user_log_predictive_density"] - baseline
            )
            metric_rows.append(row)

    comparison = pd.DataFrame(metric_rows)
    dispersion = pd.DataFrame(dispersion_rows)
    comparison.to_csv(args.output_dir / "dispersion_comparison.csv", index=False)
    dispersion.to_csv(args.output_dir / "beta_binomial_dispersion.csv", index=False)
    pd.concat(residual_frames, ignore_index=True).to_parquet(
        args.output_dir / "heldout_residuals.parquet", index=False
    )
    pd.concat(calibration_frames, ignore_index=True).to_csv(
        args.output_dir / "heldout_calibration.csv", index=False
    )
    all_residuals = pd.concat(residual_frames, ignore_index=True)
    conditional_calibration(all_residuals).to_csv(
        args.output_dir / "conditional_calibration.csv", index=False
    )
    pd.concat(parameter_frames, ignore_index=True).to_csv(
        args.output_dir / "parameter_summary.csv", index=False
    )
    metadata = {
        "day_start": args.day_start, "day_end": args.day_end,
        "train_users": len(train_users), "test_users": len(test_users),
        "vi_iterations": args.vi_iterations, "posterior_draws": args.posterior_draws,
        "seed": args.seed,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.report.write_text(render_report(comparison, dispersion, metadata), encoding="utf-8")
    print(f"Wrote comparison to {args.output_dir}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
