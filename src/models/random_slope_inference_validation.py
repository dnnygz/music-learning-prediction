#!/usr/bin/env python3
"""Validate random-slope ADVI uncertainty against NUTS on fixed users."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import spearmanr

from src.models.hierarchical_measurement import (
    RANDOM_SEED, posterior_summary, prepare_data, standardize,
)
from src.models.inference_validation import select_users
from src.models.random_slope_measurement import FORMULAS, model_priors


SCALAR_PARAMETERS = [
    "Intercept", "difficulty_z", "day_z", "1|exercise_id_sigma",
    "1|user_id_sigma", "day_z|user_id_sigma", "kappa",
]


def make_model(data: pd.DataFrame) -> bmb.Model:
    return bmb.Model(
        FORMULAS["random_slope"], data, family="beta_binomial",
        priors=model_priors("random_slope"), categorical=["exercise_id", "user_id"],
    )


def summarize_scalars(draws: dict[str, np.ndarray], method: str, response_type: str, day_std: float) -> pd.DataFrame:
    rows = []
    for parameter, parameter_draws in draws.items():
        if parameter in {"day_z", "day_z|user_id_sigma"}:
            parameter_draws = parameter_draws / day_std
            parameter = {
                "day_z": "global_slope_per_day",
                "day_z|user_id_sigma": "sigma_lambda_per_day",
            }[parameter]
        summary = posterior_summary(np.asarray(parameter_draws).reshape(-1, 1))
        rows.append({
            "response_type": response_type, "method": method, "parameter": parameter,
            **{key: float(value[0]) for key, value in summary.items()},
        })
    return pd.DataFrame(rows)


def slope_table(
    user_ids: list[str],
    global_draws: np.ndarray,
    random_draws: np.ndarray,
    day_std: float,
    method: str,
    response_type: str,
) -> pd.DataFrame:
    centered_random = random_draws - random_draws.mean(axis=1, keepdims=True)
    adjusted_global = global_draws + random_draws.mean(axis=1)
    total = (adjusted_global[:, None] + centered_random) / day_std
    summary = posterior_summary(total)
    output = pd.DataFrame({
        "user_id": user_ids, "response_type": response_type, "method": method,
        "adjusted_trajectory_slope": summary["estimate"],
        "slope_sd": summary["posterior_sd"],
        "slope_hdi_3pct": summary["hdi_3pct"],
        "slope_hdi_97pct": summary["hdi_97pct"],
        "probability_slope_positive": (total > 0).mean(axis=0),
    })
    output["slope_class"] = "uncertain"
    output.loc[output.slope_hdi_3pct > 0, "slope_class"] = "positive"
    output.loc[output.slope_hdi_97pct < 0, "slope_class"] = "negative"
    return output


def compare_slopes(advi: pd.DataFrame, nuts: pd.DataFrame, response_type: str) -> dict[str, float | str]:
    merged = advi.merge(nuts, on=["user_id", "response_type"], suffixes=("_advi", "_nuts"))
    left = merged.adjusted_trajectory_slope_advi
    right = merged.adjusted_trajectory_slope_nuts
    return {
        "response_type": response_type,
        "students": len(merged),
        "pearson_slope_correlation": float(left.corr(right)),
        "spearman_slope_correlation": float(spearmanr(left, right).statistic),
        "median_absolute_slope_difference": float(np.median(np.abs(left - right))),
        "median_advi_sd": float(merged.slope_sd_advi.median()),
        "median_nuts_sd": float(merged.slope_sd_nuts.median()),
        "median_sd_ratio_advi_over_nuts": float(
            np.median(merged.slope_sd_advi / merged.slope_sd_nuts)
        ),
        "classification_concordance": float(
            (merged.slope_class_advi == merged.slope_class_nuts).mean()
        ),
        "advi_identifiable_rate": float((merged.slope_class_advi != "uncertain").mean()),
        "nuts_identifiable_rate": float((merged.slope_class_nuts != "uncertain").mean()),
    }


def markdown(frame: pd.DataFrame) -> list[str]:
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(
            str(value) if not isinstance(value, (float, np.floating))
            else f"{value:.4f}" for value in row
        ) + " |")
    return lines


def render_report(
    scalar_comparison: pd.DataFrame,
    slope_comparison: pd.DataFrame,
    diagnostics: pd.DataFrame,
    classes: pd.DataFrame,
    metadata: dict,
) -> str:
    findings = []
    for row in slope_comparison.itertuples(index=False):
        findings.append(
            f"- {row.response_type.capitalize()}: ADVI/NUTS slope Spearman is "
            f"{row.spearman_slope_correlation:.3f}, but median ADVI uncertainty is only "
            f"{row.median_sd_ratio_advi_over_nuts:.1%} of NUTS. The identifiable rate "
            f"changes from {row.advi_identifiable_rate:.1%} to "
            f"{row.nuts_identifiable_rate:.1%}."
        )
    return "\n".join([
        "# Random-slope ADVI versus NUTS validation", "", "## Design", "",
        f"- Deterministic user subsample: {metadata['user_fraction']:.0%} ({metadata['selected_users']} users).",
        f"- ADVI: {metadata['advi_iterations']:,} iterations, {metadata['advi_draws']:,} draws.",
        f"- NUTS: {metadata['nuts_chains']} chains, {metadata['nuts_tune']:,} tune and {metadata['nuts_draws']:,} retained draws per chain.",
        "- Both methods use identical beta-binomial random-slope specifications and data scaling.",
        "", "## Population parameter comparison", "", *markdown(scalar_comparison), "",
        "## Individual slope comparison", "", *markdown(slope_comparison), "",
        "## Classification counts", "", *markdown(classes), "",
        "## NUTS diagnostics", "", *markdown(diagnostics), "",
        "## Findings", "", *findings, "",
        "ADVI is adequate for approximate ranking and point estimates, but its slope uncertainty is not calibrated. Downstream two-stage regression or classification must not treat the ADVI posterior standard deviation as ground truth. NUTS has no divergences, although remaining R-hat values above 1.01 mean this is a strong sensitivity validation rather than a final full-population posterior.",
        "",
        "The objective is uncertainty validation, not target construction. `adjusted_trajectory_slope` remains a content-adjusted platform-performance slope.",
    ]) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/student_exercise_day.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/model_outputs/random_slope_inference_v1"))
    parser.add_argument("--report", type=Path, default=Path("reports/random_slope_inference_v1.md"))
    parser.add_argument("--user-fraction", type=float, default=0.25)
    parser.add_argument("--advi-iterations", type=int, default=20000)
    parser.add_argument("--advi-draws", type=int, default=1000)
    parser.add_argument("--nuts-tune", type=int, default=750)
    parser.add_argument("--nuts-draws", type=int, default=750)
    parser.add_argument("--nuts-chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    data = prepare_data(pd.read_parquet(args.input), 7, 31)
    subset, selected_users = select_users(data, args.user_fraction, args.seed)
    scalar_frames, slope_frames, slope_comparisons, diagnostic_rows = [], [], [], []

    for response_index, response_type in enumerate(("note", "chord")):
        frame = subset.loc[subset.response_type == response_type].copy()
        frame, _, scaling = standardize(frame, frame.iloc[:0].copy())
        day_std = scaling["day_index_std"]

        print(f"Fitting {response_type} random-slope ADVI on {len(frame):,} rows", flush=True)
        advi_model = make_model(frame)
        approximation = advi_model.fit(
            inference_method="vi", n=args.advi_iterations, method="advi",
            obj_optimizer=pm.adam(learning_rate=0.01), progressbar=False,
            random_seed=args.seed + response_index,
        )
        advi_trace = approximation.sample(
            args.advi_draws, return_inferencedata=False,
            random_seed=args.seed + response_index,
        )
        advi_scalar = {
            name: advi_trace.get_values(name, combine=True) for name in SCALAR_PARAMETERS
        }
        scalar_frames.append(summarize_scalars(advi_scalar, "advi", response_type, day_std))
        user_ids = [str(x) for x in advi_model.backend.model.coords["user_id__factor_dim"]]
        advi_slopes = slope_table(
            user_ids,
            advi_trace.get_values("day_z", combine=True),
            advi_trace.get_values("day_z|user_id", combine=True),
            day_std, "advi", response_type,
        )
        slope_frames.append(advi_slopes)

        print(f"Fitting {response_type} random-slope NUTS on {len(frame):,} rows", flush=True)
        nuts_model = make_model(frame)
        idata = nuts_model.fit(
            draws=args.nuts_draws, tune=args.nuts_tune, chains=args.nuts_chains,
            cores=args.nuts_chains,
            random_seed=[
                args.seed + 100 + response_index * 10 + chain
                for chain in range(args.nuts_chains)
            ], nuts={"target_accept": 0.9}, progressbar=False,
        )
        nuts_scalar = {
            name: idata.posterior[name].values.reshape(-1) for name in SCALAR_PARAMETERS
        }
        scalar_frames.append(summarize_scalars(nuts_scalar, "nuts", response_type, day_std))
        nuts_random = idata.posterior["day_z|user_id"].values.reshape(
            -1, len(user_ids)
        )
        nuts_slopes = slope_table(
            user_ids, idata.posterior["day_z"].values.reshape(-1), nuts_random,
            day_std, "nuts", response_type,
        )
        slope_frames.append(nuts_slopes)
        slope_comparisons.append(compare_slopes(advi_slopes, nuts_slopes, response_type))

        diagnostics = az.summary(idata, var_names=SCALAR_PARAMETERS, kind="diagnostics")
        diagnostic_rows.append({
            "response_type": response_type,
            "divergences": int(idata.sample_stats.diverging.sum().item()),
            "max_rhat": float(diagnostics.r_hat.max()),
            "min_ess_bulk": float(diagnostics.ess_bulk.min()),
            "min_ess_tail": float(diagnostics.ess_tail.min()),
        })

    scalar = pd.concat(scalar_frames, ignore_index=True)
    slopes = pd.concat(slope_frames, ignore_index=True)
    slope_comparison = pd.DataFrame(slope_comparisons)
    diagnostics = pd.DataFrame(diagnostic_rows)
    classes = (
        slopes.groupby(["response_type", "method", "slope_class"], observed=True)
        .size().rename("students").reset_index()
    )
    totals = classes.groupby(["response_type", "method"]).students.transform("sum")
    classes["percentage"] = classes.students / totals
    wide = scalar.pivot(index=["response_type", "parameter"], columns="method")
    scalar_comparison = pd.DataFrame({
        "advi_estimate": wide[("estimate", "advi")],
        "nuts_estimate": wide[("estimate", "nuts")],
        "advi_sd": wide[("posterior_sd", "advi")],
        "nuts_sd": wide[("posterior_sd", "nuts")],
    }).reset_index()
    scalar_comparison["absolute_difference"] = (
        scalar_comparison.advi_estimate - scalar_comparison.nuts_estimate
    ).abs()
    scalar_comparison["difference_in_nuts_sd"] = (
        scalar_comparison.absolute_difference / scalar_comparison.nuts_sd.replace(0, np.nan)
    )

    scalar.to_csv(args.output_dir / "posterior_summaries.csv", index=False)
    scalar_comparison.to_csv(args.output_dir / "population_parameter_comparison.csv", index=False)
    slopes.to_parquet(args.output_dir / "student_slope_comparison.parquet", index=False)
    slope_comparison.to_csv(args.output_dir / "individual_slope_agreement.csv", index=False)
    classes.to_csv(args.output_dir / "classification_comparison.csv", index=False)
    diagnostics.to_csv(args.output_dir / "nuts_diagnostics.csv", index=False)
    metadata = {
        "user_fraction": args.user_fraction, "selected_users": len(selected_users),
        "advi_iterations": args.advi_iterations, "advi_draws": args.advi_draws,
        "nuts_tune": args.nuts_tune, "nuts_draws": args.nuts_draws,
        "nuts_chains": args.nuts_chains, "seed": args.seed,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.report.write_text(
        render_report(scalar_comparison, slope_comparison, diagnostics, classes, metadata),
        encoding="utf-8",
    )
    print(f"Wrote random-slope inference validation to {args.output_dir}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
