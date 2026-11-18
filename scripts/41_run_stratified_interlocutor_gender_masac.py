#!/usr/bin/env python3
"""
Stratified interlocutor-gender prosody models (MaSaC).

Subsets by speaker gender and age_bucket (Younger / Older), then fits
mixed-gender vs same-gender models on precomputed interlocutor labels.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def _select_numeric_features(df: pd.DataFrame, n: int) -> List[str]:
    exclude = {
        "start",
        "end",
        "start_time",
        "end_time",
        "duration_sec",
        "n_switches",
        "n_discourse_markers",
        "n_repetitions",
        "audience_l2_ratio",
    }
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    candidates = [c for c in numeric if c not in exclude]
    if not candidates:
        return []
    var = df[candidates].var(numeric_only=True).sort_values(ascending=False)
    return var.head(n).index.tolist()


def _lhs(col: str) -> str:
    if col.isidentifier():
        return col
    esc = str(col).replace("\\", "\\\\").replace('"', '\\"')
    return f'Q("{esc}")'


def _fit_models(df: pd.DataFrame, outcomes: List[str], group_var: str) -> pd.DataFrame:
    rows = []
    for y in outcomes:
        cols = [y, "gender_match", "condition", group_var]
        if "duration_sec" in df.columns:
            cols.append("duration_sec")
        d = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) < 200:
            rows.append({"feature": y, "success": False, "error": "insufficient_data"})
            continue
        d["gender_match"] = pd.Categorical(d["gender_match"].astype(str), ["same", "different", "unknown"])
        rhs = "C(gender_match, Treatment(reference='same')) + C(condition)"
        if "duration_sec" in cols:
            rhs += " + duration_sec"
        formula = f"{_lhs(y)} ~ {rhs}"
        try:
            res = smf.ols(formula, d).fit(cov_type="cluster", cov_kwds={"groups": d[group_var]})
        except Exception as e:
            rows.append({"feature": y, "success": False, "error": str(e)})
            continue
        key = "C(gender_match, Treatment(reference='same'))[T.different]"
        rows.append(
            {
                "feature": y,
                "success": True,
                "n_obs": len(d),
                "n_groups": int(d[group_var].nunique()),
                "coef_diff_vs_same": float(res.params.get(key, np.nan)),
                "p_diff_vs_same": float(res.pvalues.get(key, np.nan)),
            }
        )
    return pd.DataFrame(rows)


def _run_stratum(df: pd.DataFrame, out_dir: pathlib.Path, n_features: int, group_var: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes = _select_numeric_features(df, n_features)
    models = _fit_models(df, outcomes, group_var=group_var)
    models.to_csv(out_dir / "masac_interlocutor_gender_models.csv", index=False)
    df["gender_match"].value_counts(dropna=False).rename_axis("gender_match").reset_index(name="count").to_csv(
        out_dir / "gender_match_counts.csv", index=False
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Run stratified MaSaC interlocutor-gender models")
    p.add_argument(
        "--input",
        type=str,
        default="results/interlocutor_gender/masac_dyad/masac_integrated_features_dyad.csv",
    )
    p.add_argument(
        "--labels",
        type=str,
        default="results/interlocutor_gender/masac_dyad/models/seame_interlocutor_gender_labels.csv",
    )
    p.add_argument(
        "--out-root",
        type=str,
        default="results/interlocutor_gender/masac_stratified/dyad",
    )
    p.add_argument("--n-features", type=int, default=20)
    p.add_argument("--group-var", type=str, default="speaker")
    args = p.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    labels = pd.read_csv(args.labels, usecols=["utt_id", "gender_match"])
    df = df.merge(labels, on="utt_id", how="left", validate="1:1")
    df = df[df["gender_match"].isin(["same", "different", "unknown"])].copy()

    out_root = pathlib.Path(args.out_root)
    strata = {
        "gender=female": df[df["gender"].astype(str).str.lower() == "female"],
        "gender=male": df[df["gender"].astype(str).str.lower() == "male"],
        "age_bucket=Younger": df[df["age_bucket"].astype(str) == "Younger"],
        "age_bucket=Older": df[df["age_bucket"].astype(str) == "Older"],
        "gender=female_age=Younger": df[
            (df["gender"].astype(str).str.lower() == "female") & (df["age_bucket"].astype(str) == "Younger")
        ],
        "gender=female_age=Older": df[
            (df["gender"].astype(str).str.lower() == "female") & (df["age_bucket"].astype(str) == "Older")
        ],
        "gender=male_age=Younger": df[
            (df["gender"].astype(str).str.lower() == "male") & (df["age_bucket"].astype(str) == "Younger")
        ],
    }
    for name, sub in strata.items():
        if len(sub) < 200:
            continue
        _run_stratum(sub, out_root / name, int(args.n_features), str(args.group_var))
    print(f"Wrote stratified models under {out_root}")


if __name__ == "__main__":
    main()
