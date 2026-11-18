#!/usr/bin/env python3
"""Appendix: primary mixed-effects models with duration_sec as a covariate."""

import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

import importlib.util
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "mixed_effects_analysis", _ROOT / "scripts" / "18_mixed_effects_analysis.py"
)
_mem = importlib.util.module_from_spec(_spec)
sys.modules["mixed_effects_analysis"] = _mem
_spec.loader.exec_module(_mem)
formula_lhs = _mem.formula_lhs
merge_features_and_manifest = _mem.merge_features_and_manifest
select_key_features = _mem.select_key_features


def fit_duration_controlled(df: pd.DataFrame, feature: str, group_var: str) -> dict:
    cols = [feature, "condition", "has_discourse_marker", "has_repetition", "duration_sec", group_var]
    model_df = df[cols].dropna()
    if len(model_df) < 20 or model_df[group_var].nunique() < 2:
        return {"success": False, "feature": feature, "error": "insufficient data"}

    formula = (
        f"{formula_lhs(feature)} ~ C(condition) + has_discourse_marker + has_repetition + duration_sec"
    )
    try:
        result = smf.mixedlm(formula, model_df, groups=model_df[group_var]).fit()
    except Exception:
        try:
            result = smf.mixedlm(formula, model_df, groups=model_df[group_var]).fit(method="lbfgs")
        except Exception as exc:
            return {"success": False, "feature": feature, "error": str(exc)}

    row = {
        "success": True,
        "feature": feature,
        "formula": formula,
        "n_obs": int(result.nobs),
        "n_groups": int(model_df[group_var].nunique()),
    }
    for name, coef in result.params.items():
        row[f"coef::{name}"] = float(coef)
        row[f"p::{name}"] = float(result.pvalues.get(name, np.nan))
    return row


def run(integrated_csv: str, feature_csv: str, output_csv: str, n_features: int = 10) -> pd.DataFrame:
    feat_df = pd.read_csv(feature_csv, low_memory=False)
    df = pd.read_csv(integrated_csv, low_memory=False)
    key_features = select_key_features(feat_df, n_features=n_features)
    if "duration_sec" not in df.columns:
        raise ValueError("duration_sec missing from integrated manifest")

    group_var = "speaker_id" if "speaker_id" in df.columns else "speaker"
    features = key_features
    rows = [fit_duration_controlled(df, f, group_var) for f in features]
    out = pd.DataFrame([r for r in rows if r.get("success")])
    out_path = pathlib.Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} models)")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Duration-controlled mixed-effects sensitivity")
    p.add_argument("--corpus", choices=["seame", "masac"], required=True)
    p.add_argument("--n-features", type=int, default=10)
    args = p.parse_args()
    integrated = f"results/functional_anchors/{args.corpus}/{args.corpus}_integrated_features.csv"
    features = f"features/{args.corpus}_disvoice_utt.csv"
    output = f"results/mixed_effects/{args.corpus}_full/{args.corpus}_duration_controlled_mem.csv"
    run(integrated, features, output, n_features=args.n_features)
