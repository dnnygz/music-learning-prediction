#!/usr/bin/env python3
"""Validate beta-binomial ADVI against NUTS on a fixed user subsample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm

from src.models.hierarchical_measurement import (
    RANDOM_SEED, formula_for, posterior_summary, prepare_data, priors_for, standardize,
)


PARAMETERS = [
    "Intercept", "difficulty_z", "day_z", "1|exercise_id_sigma",
    "1|user_id_sigma", "kappa",
]


def make_model(data: pd.DataFrame) -> bmb.Model:
    priors = priors_for("model_4")
    priors["kappa"] = bmb.Prior("HalfNormal", sigma=50)
    return bmb.Model(
        formula_for("model_4"), data, family="beta_binomial", priors=priors,
        categorical=["exercise_id", "user_id"],
    )


def select_users(data: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.DataFrame, list[str]]:
    users = np.array(sorted(data["user_id"].unique()))
    rng = np.random.default_rng(seed)
    selected = sorted(rng.choice(users, size=max(2, round(len(users) * fraction)), replace=False))
    return data.loc[data["user_id"].isin(selected)].copy(), selected


def summarize_draws(
    draws_by_parameter: dict[str, np.ndarray], method: str, response_type: str
) -> pd.DataFrame:
    rows = []
    for parameter, draws in draws_by_parameter.items():
        summary = posterior_summary(np.asarray(draws).reshape(-1, 1))
        rows.append({
            "response_type": response_type, "method": method, "parameter": parameter,
            **{key: float(value[0]) for key, value in summary.items()},
        })
    return pd.DataFrame(rows)


def render_report(comparison: pd.DataFrame, diagnostics: pd.DataFrame, metadata: dict) -> str:
    columns = [
        "response_type", "parameter", "advi_estimate", "nuts_estimate",
        "absolute_difference", "difference_in_nuts_sd",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in comparison[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(
            str(value) if not isinstance(value, (float, np.floating)) else f"{value:.4f}"
            for value in row
        ) + " |")
    diagnostic_lines = []
    for row in diagnostics.itertuples(index=False):
        diagnostic_lines.append(
            f"- {row.response_type.capitalize()}: {row.divergences} divergences; "
            f"maximum R-hat {row.max_rhat:.3f}; minimum bulk ESS {row.min_ess_bulk:.0f}."
        )
    return "\n".join([
        "# Beta-binomial inference validation", "", "## Design", "",
        f"- Deterministic user subsample: {metadata['user_fraction']:.0%}.",
        f"- ADVI: {metadata['advi_iterations']:,} iterations and {metadata['advi_draws']:,} posterior draws.",
        f"- NUTS: {metadata['nuts_chains']} chains, {metadata['nuts_tune']:,} tune and {metadata['nuts_draws']:,} retained draws per chain.",
        "- Both methods use the same rows, scaling, likelihood, priors, and formula.",
        "", "## Parameter comparison", "", *lines, "", "## NUTS diagnostics", "",
        *diagnostic_lines, "",
        "Agreement is evaluated on the global coefficients, hierarchical standard deviations, and beta-binomial concentration. Group-level effects themselves are not compared one by one in this screening run.",
    ]) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/student_exercise_day.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/model_outputs/inference_validation_v1"))
    parser.add_argument("--report", type=Path, default=Path("reports/inference_validation_v1.md"))
    parser.add_argument("--user-fraction", type=float, default=0.25)
    parser.add_argument("--advi-iterations", type=int, default=20000)
    parser.add_argument("--advi-draws", type=int, default=1000)
    parser.add_argument("--nuts-tune", type=int, default=500)
    parser.add_argument("--nuts-draws", type=int, default=500)
    parser.add_argument("--nuts-chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    data = prepare_data(pd.read_parquet(args.input), 7, 31)
    subset, selected_users = select_users(data, args.user_fraction, args.seed)
    summaries, diagnostic_rows = [], []

    for response_index, response_type in enumerate(("note", "chord")):
        frame = subset.loc[subset["response_type"] == response_type].copy()
        frame, _, _ = standardize(frame, frame.iloc[:0].copy())
        print(f"Fitting {response_type} ADVI on {len(frame):,} rows", flush=True)
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
        advi_draws = {name: advi_trace.get_values(name, combine=True) for name in PARAMETERS}
        summaries.append(summarize_draws(advi_draws, "advi", response_type))

        print(f"Fitting {response_type} NUTS on {len(frame):,} rows", flush=True)
        nuts_model = make_model(frame)
        idata = nuts_model.fit(
            draws=args.nuts_draws, tune=args.nuts_tune, chains=args.nuts_chains,
            cores=args.nuts_chains, random_seed=[
                args.seed + 100 + response_index * 10 + chain
                for chain in range(args.nuts_chains)
            ], nuts={"target_accept": 0.9}, progressbar=False,
        )
        nuts_draws = {
            name: idata.posterior[name].values.reshape(-1) for name in PARAMETERS
        }
        summaries.append(summarize_draws(nuts_draws, "nuts", response_type))
        diagnostic = az.summary(idata, var_names=PARAMETERS, kind="diagnostics")
        divergences = int(idata.sample_stats["diverging"].sum().item())
        diagnostic_rows.append({
            "response_type": response_type, "divergences": divergences,
            "max_rhat": float(diagnostic["r_hat"].max()),
            "min_ess_bulk": float(diagnostic["ess_bulk"].min()),
            "min_ess_tail": float(diagnostic["ess_tail"].min()),
        })

    summary = pd.concat(summaries, ignore_index=True)
    wide = summary.pivot(index=["response_type", "parameter"], columns="method")
    comparison = pd.DataFrame({
        "advi_estimate": wide[("estimate", "advi")],
        "nuts_estimate": wide[("estimate", "nuts")],
        "advi_posterior_sd": wide[("posterior_sd", "advi")],
        "nuts_posterior_sd": wide[("posterior_sd", "nuts")],
    }).reset_index()
    comparison["absolute_difference"] = (
        comparison["advi_estimate"] - comparison["nuts_estimate"]
    ).abs()
    comparison["difference_in_nuts_sd"] = comparison["absolute_difference"] / comparison[
        "nuts_posterior_sd"
    ].replace(0, np.nan)
    diagnostics = pd.DataFrame(diagnostic_rows)
    summary.to_csv(args.output_dir / "posterior_summaries.csv", index=False)
    comparison.to_csv(args.output_dir / "advi_nuts_comparison.csv", index=False)
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
    args.report.write_text(render_report(comparison, diagnostics, metadata), encoding="utf-8")
    print(f"Wrote inference validation to {args.output_dir}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
