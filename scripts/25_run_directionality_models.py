#!/usr/bin/env python3
"""
Fit CS-only directionality models alongside all-data condition models.

Two model families (per prosodic feature):
1. All-data condition model:
   Y ~ condition + has_discourse_marker + has_repetition

2. CS-only directionality model:
   Y ~ is_l1_to_l2 + has_discourse_marker + has_repetition
       + is_l1_to_l2:has_discourse_marker

For SEAME, these can be extended to mixed-effects models with speaker
random intercepts in a later iteration. For now, this script uses OLS
for robustness and speed, and outputs coefficient tables.
"""

import argparse
import pathlib
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def formula_lhs(column_name: str) -> str:
    """Quote dependent variable for patsy when not a valid Python identifier (e.g. 1F0max)."""
    if column_name.isidentifier():
        return column_name
    esc = column_name.replace("\\", "\\\\").replace('"', '\\"')
    return f'Q("{esc}")'


def select_prosodic_features(df: pd.DataFrame, n_features: int = 10) -> List[str]:
    """Select a set of prosodic features to model."""
    candidates = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and any(x in c.lower() for x in ["f0", "pitch", "duration", "rate", "pause", "energy", "intensity"])
    ]
    if not candidates:
        return []
    variances = df[candidates].var().sort_values(ascending=False)
    return variances.head(n_features).index.tolist()


def fit_ols(formula: str, data: pd.DataFrame) -> Dict:
    """Fit an OLS model and return summary statistics."""
    try:
        model = smf.ols(formula, data=data)
        res = model.fit()
        return {
            "success": True,
            "formula": formula,
            "params": res.params.to_dict(),
            "pvalues": res.pvalues.to_dict(),
            "aic": res.aic,
            "bic": res.bic,
            "n": int(res.nobs),
        }
    except Exception as e:
        return {"success": False, "error": str(e), "formula": formula}


def choose_covariates(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    exclude = set(exclude or [])
    covariates = []
    for col in ["has_discourse_marker", "has_repetition"]:
        if col in df.columns and col not in exclude:
            covariates.append(col)
    for col in ["gender", "age_bucket", "nationality", "dialogue_act", "audience_proficiency_proxy"]:
        if col in df.columns and col not in exclude:
            n_levels = df[col].dropna().astype(str).nunique()
            if 1 < n_levels <= 20:
                covariates.append(col)
    return covariates


def term_for_column(df: pd.DataFrame, col: str) -> str:
    if col in df.columns and (
        str(df[col].dtype) == "object" or str(df[col].dtype).startswith("category")
    ):
        return f"C({col})"
    return col


def fit_cs_predictor_models(
    cs_df: pd.DataFrame,
    features: List[str],
    predictor: str,
    model_type: str,
    covariate_exclude: Optional[List[str]] = None,
) -> List[Dict]:
    """Fit CS-only OLS models with a single primary predictor."""
    rows = []
    for feat in features:
        covars = choose_covariates(cs_df, exclude=covariate_exclude or [])
        if predictor == "C(cs_strategy)":
            subset = cs_df[cs_df["cs_strategy"].isin(["insertional", "alternational"])].copy()
            predictors = ["C(cs_strategy)"] + covars
        else:
            subset = cs_df.copy()
            predictors = [predictor] + covars

        cols = [feat, predictor if predictor != "C(cs_strategy)" else "cs_strategy"] + covars
        cols = [c for c in cols if c in subset.columns]
        data = subset[cols].dropna()
        if predictor == "C(cs_strategy)":
            data = data[data["cs_strategy"].isin(["insertional", "alternational"])]

        if len(data) <= 10:
            rows.append({
                "success": False,
                "error": "Insufficient CS data",
                "feature": feat,
                "model_type": model_type,
            })
            continue

        if predictor != "C(cs_strategy)" and data[predictor].nunique() <= 1:
            rows.append({
                "success": False,
                "error": f"No variability in {predictor}",
                "feature": feat,
                "model_type": model_type,
            })
            continue

        predictor_terms = []
        for p in predictors:
            if p == "C(cs_strategy)":
                predictor_terms.append(p)
            else:
                predictor_terms.append(term_for_column(data, p))

        lhs = formula_lhs(feat)
        formula = f"{lhs} ~ " + " + ".join(predictor_terms)
        res = fit_ols(formula, data)
        res["feature"] = feat
        res["model_type"] = model_type
        rows.append(res)
    return rows


def run_directionality_models(integrated_file: str, output_dir: str, corpus: str,
                              n_features: int = 10) -> None:
    """Run condition and CS-only directionality models and save results."""
    print(f"Loading integrated features from: {integrated_file}")
    df = pd.read_csv(integrated_file, low_memory=False)
    print(f"Total rows: {len(df)}")

    # Basic checks
    required_cols = ["condition", "lang"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in integrated dataset")

    # Prepare CS-only view with directionality
    cs_df = df[df["lang"] == "CS"].copy()
    if "switch_direction_class" in cs_df.columns:
        cs_df["is_l1_to_l2"] = (cs_df["switch_direction_class"] == "L1→L2").astype(int)
    else:
        cs_df["is_l1_to_l2"] = np.nan

    # Choose features
    features = select_prosodic_features(df, n_features=n_features)
    print(f"Selected {len(features)} prosodic features: {', '.join(features)}")

    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_data_rows = []
    cs_only_rows = []
    cs_quantity_rows = []
    cs_mi_rows = []
    cs_strategy_rows = []

    for feat in features:
        # All-data condition model
        covars = choose_covariates(df)
        predictors = ["condition"] + covars
        predictor_terms = [term_for_column(df, c) for c in predictors]
        lhs = formula_lhs(feat)
        interaction_terms = []
        for col in ["gender", "age_bucket", "nationality", "dialogue_act", "audience_proficiency_proxy"]:
            if col in covars:
                interaction_terms.append(f"C(condition):C({col})")
        formula_all = f"{lhs} ~ " + " + ".join(predictor_terms + interaction_terms)
        res_all = fit_ols(formula_all, df[[feat] + predictors].dropna())
        res_all["feature"] = feat
        res_all["model_type"] = "all_condition"
        all_data_rows.append(res_all)

        # CS-only directionality model
        cs_subset = cs_df[[feat, "is_l1_to_l2"] + predictors[1:]].dropna()
        if len(cs_subset) > 10 and cs_subset["is_l1_to_l2"].nunique() > 1:
            dir_covars = choose_covariates(cs_subset)
            dir_predictors = ["is_l1_to_l2"] + dir_covars

            # Add interaction is_l1_to_l2:has_discourse_marker if both present
            interaction_terms = []
            if "has_discourse_marker" in dir_predictors:
                interaction_terms.append("is_l1_to_l2:has_discourse_marker")
            for col in ["gender", "age_bucket", "nationality", "dialogue_act", "audience_proficiency_proxy"]:
                if col in dir_covars:
                    interaction_terms.append(f"is_l1_to_l2:C({col})")

            dir_predictor_terms = [term_for_column(cs_subset, c) for c in dir_predictors]
            formula_cs = f"{lhs} ~ " + " + ".join(dir_predictor_terms + interaction_terms)
            res_cs = fit_ols(formula_cs, cs_subset)
            res_cs["feature"] = feat
            res_cs["model_type"] = "cs_directionality"
            cs_only_rows.append(res_cs)
        else:
            cs_only_rows.append({
                "success": False,
                "error": "Insufficient CS data or no direction variability",
                "feature": feat,
                "model_type": "cs_directionality"
            })

    cs_quantity_rows = fit_cs_predictor_models(
        cs_df, features, "n_switches", "cs_quantity_n_switches"
    )
    cs_mi_rows = fit_cs_predictor_models(
        cs_df, features, "m_index", "cs_quantity_m_index"
    )
    cs_mi_rows.extend(
        fit_cs_predictor_models(cs_df, features, "i_index", "cs_quantity_i_index")
    )
    if "cs_strategy" in cs_df.columns:
        cs_strategy_rows = fit_cs_predictor_models(
            cs_df,
            features,
            "C(cs_strategy)",
            "cs_strategy",
            covariate_exclude=["dialogue_act"],
        )

    # Save results
    all_df = pd.DataFrame(all_data_rows)
    cs_df_res = pd.DataFrame(cs_only_rows)
    cs_quantity_df = pd.DataFrame(cs_quantity_rows)
    cs_mi_df = pd.DataFrame(cs_mi_rows)
    cs_strategy_df = pd.DataFrame(cs_strategy_rows)

    all_out = output_path / f"{corpus}_condition_models_ols.csv"
    cs_out = output_path / f"{corpus}_cs_directionality_models_ols.csv"
    quantity_out = output_path / f"{corpus}_cs_quantity_models_ols.csv"
    mi_out = output_path / f"{corpus}_cs_mi_index_models_ols.csv"
    strategy_out = output_path / f"{corpus}_cs_strategy_models_ols.csv"
    all_df.to_csv(all_out, index=False)
    cs_df_res.to_csv(cs_out, index=False)
    cs_quantity_df.to_csv(quantity_out, index=False)
    cs_mi_df.to_csv(mi_out, index=False)
    cs_strategy_df.to_csv(strategy_out, index=False)

    print(f"\nSaved all-data condition models to: {all_out}")
    print(f"Saved CS-only directionality models to: {cs_out}")
    print(f"Saved CS quantity (n_switches) models to: {quantity_out}")
    print(f"Saved CS M/I-index models to: {mi_out}")
    if len(cs_strategy_df):
        print(f"Saved CS strategy models to: {strategy_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run condition and CS-only directionality models (OLS)")
    p.add_argument("--integrated", type=str, required=True,
                   help="Integrated features CSV (from 19_integrate_features_for_analysis.py)")
    p.add_argument("--output-dir", type=str, default="results/directionality_models",
                   help="Output directory")
    p.add_argument("--corpus", type=str, choices=["seame", "masac"], required=True,
                   help="Corpus name")
    p.add_argument("--n-features", type=int, default=10,
                   help="Number of features to model")

    args = p.parse_args()

    run_directionality_models(args.integrated, args.output_dir, args.corpus, args.n_features)

