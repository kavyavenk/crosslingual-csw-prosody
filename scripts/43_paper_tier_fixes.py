#!/usr/bin/env python3
"""
Tier 1 + Tier 2 paper fixes: unified feature selection, Table 4 export,
RQ3 condition×interlocutor models, Cohen's d / cross-corpus comparison,
speaker-level t-test robustness note.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind
import statsmodels.stats.multitest as smm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paper_feature_selection import (
    disvoice_category,
    primary_outcome_list,
    select_primary_outcomes,
)

# Re-use interlocutor labeling from script 30
import importlib.util
_inter_path = pathlib.Path(__file__).resolve().parent / "30_interlocutor_gender_effects.py"
_spec = importlib.util.spec_from_file_location("interlocutor_gender_effects", _inter_path)
_interlocutor = importlib.util.module_from_spec(_spec)
sys.modules["interlocutor_gender_effects"] = _interlocutor
_spec.loader.exec_module(_interlocutor)
add_interlocutor_gender = _interlocutor.add_interlocutor_gender


def formula_lhs(col: str) -> str:
    if col.isidentifier():
        return col
    esc = str(col).replace("\\", "\\\\").replace('"', '\\"')
    return f'Q("{esc}")'


def fit_primary_mem(
    df: pd.DataFrame, feature: str, group_col: str = "speaker_id"
) -> Optional[Dict]:
    cols = [feature, "condition", "has_discourse_marker", "has_repetition", group_col]
    m = df[cols].dropna().copy()
    if len(m) < 100:
        return None
    formula = (
        f"{formula_lhs(feature)} ~ C(condition) + has_discourse_marker "
        f"+ has_repetition"
    )
    try:
        res = smf.mixedlm(formula, m, groups=m[group_col]).fit(method=["lbfgs", "powell"])
    except Exception:
        try:
            res = smf.mixedlm(formula, m, groups=m[group_col]).fit()
        except Exception as e:
            return {"error": str(e)}
    params = res.params
    pvals = res.pvalues
    l1_key = "C(condition)[T.monolingual_ZH]"
    if l1_key not in params and "C(condition)[T.monolingual_HI]" in params:
        l1_key = "C(condition)[T.monolingual_HI]"
    re_var = float(res.cov_re.iloc[0, 0]) if hasattr(res, "cov_re") else np.nan
    re_p = np.nan
    if hasattr(res, "cov_re") and hasattr(res, "pvalues_re"):
        try:
            re_p = float(res.pvalues_re.iloc[0, 0])
        except Exception:
            pass
    return {
        "feature": feature,
        "category": disvoice_category(feature),
        "n_obs": len(m),
        "beta_mono_en": float(params.get("C(condition)[T.monolingual_EN]", np.nan)),
        "p_mono_en": float(pvals.get("C(condition)[T.monolingual_EN]", np.nan)),
        "beta_mono_l1": float(params.get(l1_key, np.nan)),
        "p_mono_l1": float(pvals.get(l1_key, np.nan)),
        "beta_dm": float(params.get("has_discourse_marker[T.True]", np.nan)),
        "p_dm": float(pvals.get("has_discourse_marker[T.True]", np.nan)),
        "beta_rep": float(params.get("has_repetition[T.True]", np.nan)),
        "p_rep": float(pvals.get("has_repetition[T.True]", np.nan)),
        "speaker_re_var": re_var,
        "speaker_re_p": re_p,
    }


def fit_condition_gender_interaction(
    df: pd.DataFrame, feature: str, group_col: str = "speaker_id"
) -> Optional[Dict]:
    """RQ3: does CS–mono gap differ by interlocutor gender_match?"""
    cols = [
        feature,
        "condition",
        "gender_match",
        "has_discourse_marker",
        "has_repetition",
        group_col,
    ]
    if "duration_sec" in df.columns:
        cols.append("duration_sec")
    m = df[cols].dropna().copy()
    m = m[m["gender_match"].isin(["same", "different"])]
    if len(m) < 500:
        return None
    m["gender_match"] = pd.Categorical(m["gender_match"], categories=["same", "different"])
    dur_term = " + duration_sec" if "duration_sec" in m.columns else ""
    formula = (
        f"{formula_lhs(feature)} ~ C(condition) * C(gender_match, Treatment(reference='same')) "
        f"+ has_discourse_marker + has_repetition{dur_term}"
    )
    try:
        res = smf.mixedlm(formula, m, groups=m[group_col]).fit(method=["lbfgs", "powell"])
    except Exception:
        try:
            res = smf.mixedlm(formula, m, groups=m[group_col]).fit()
        except Exception as e:
            return {"feature": feature, "error": str(e)}
    params = res.params
    pvals = res.pvalues
    l1 = "C(condition)[T.monolingual_ZH]"
    if l1 not in params:
        l1 = "C(condition)[T.monolingual_HI]"
    keys = {
        "beta_mono_en": "C(condition)[T.monolingual_EN]",
        "beta_mono_l1": l1,
        "beta_gender_diff": "C(gender_match, Treatment(reference='same'))[T.different]",
        "beta_en_x_gender": (
            "C(condition)[T.monolingual_EN]:C(gender_match, Treatment(reference='same'))[T.different]"
        ),
        "beta_l1_x_gender": (
            f"{l1}:C(gender_match, Treatment(reference='same'))[T.different]"
        ),
    }
    out = {"feature": feature, "category": disvoice_category(feature), "n_obs": len(m)}
    for name, key in keys.items():
        out[name] = float(params.get(key, np.nan))
        out[name.replace("beta_", "p_")] = float(pvals.get(key, np.nan))
    return out


def speaker_level_ttest(
    df: pd.DataFrame, feature: str, condition_col: str = "condition"
) -> Tuple[float, float]:
    """One mean per speaker per condition; Welch t-test CS vs mono EN."""
    sp = "speaker_id" if "speaker_id" in df.columns else "speaker"
    sub = df[[sp, condition_col, feature]].dropna()
    cs = sub[sub[condition_col] == "code_switched"].groupby(sp)[feature].mean()
    en = sub[sub[condition_col] == "monolingual_EN"].groupby(sp)[feature].mean()
    common = cs.index.intersection(en.index)
    if len(common) < 10:
        return np.nan, np.nan
    a, b = cs.loc[common].values, en.loc[common].values
    t, p = ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def category_cohen_summary(contrast_csv: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(contrast_csv)
    df["category"] = df["feature"].map(disvoice_category)
    rows = []
    for cat, g in df.groupby("category"):
        rows.append(
            {
                "category": cat,
                "n_features": len(g),
                "n_sig": int(g["sig"].sum()),
                "mean_abs_d": float(g["cohen_d"].abs().mean()),
                "max_abs_d": float(g["cohen_d"].abs().max()),
            }
        )
    return pd.DataFrame(rows)


def export_table4(corpus: str, df: pd.DataFrame, out_dir: pathlib.Path) -> pd.DataFrame:
    group = "speaker_id" if "speaker_id" in df.columns else "speaker"
    features = primary_outcome_list(df, k_per_category=3)
    rows = []
    for feat in features:
        r = fit_primary_mem(df, feat, group)
        if r and "error" not in r:
            r["corpus"] = corpus
            rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / f"table4_mem_primary_{corpus}.csv", index=False)
    return tab


def run_rq3(corpus: str, df: pd.DataFrame, out_dir: pathlib.Path, dyad_only: bool = False) -> pd.DataFrame:
    order_col = None
    if "start_time" not in df.columns and "utt_id" in df.columns:
        parts = df["utt_id"].astype(str).str.rsplit("_", n=1, expand=True)
        if parts.shape[1] == 2:
            df = df.copy()
            df["_utt_ord"] = pd.to_numeric(parts[1], errors="coerce")
            order_col = "_utt_ord"
    interaction_col = "interaction_id"
    if interaction_col not in df.columns or df[interaction_col].isna().all():
        interaction_col = "file_id"
    df = add_interlocutor_gender(df, interaction_col=interaction_col, order_col=order_col)
    m = df[df["prev_speaker"].notna()].copy()
    if dyad_only:
        sp_col = "speaker_id" if "speaker_id" in m.columns else "speaker"
        sizes = m.groupby(interaction_col)[sp_col].nunique()
        dyad_ids = sizes[sizes == 2].index
        m = m[m[interaction_col].isin(dyad_ids)]
    m = m[m["gender_match"].isin(["same", "different"])]
    if m.empty:
        pd.DataFrame().to_csv(
            out_dir / f"rq3_condition_x_gender_{corpus}{'_dyad' if dyad_only else '_full'}.csv",
            index=False,
        )
        return pd.DataFrame()
    features = primary_outcome_list(m, k_per_category=3)
    group = "speaker_id" if "speaker_id" in m.columns else "speaker"
    rows = []
    for feat in features:
        r = fit_condition_gender_interaction(m, feat, group)
        if r:
            r["corpus"] = corpus
            r["dyad_only"] = dyad_only
            rows.append(r)
    tab = pd.DataFrame(rows)
    suffix = "_dyad" if dyad_only else "_full"
    tab.to_csv(out_dir / f"rq3_condition_x_gender_{corpus}{suffix}.csv", index=False)
    return tab


def selection_manifest(df: pd.DataFrame, corpus: str, out_dir: pathlib.Path) -> None:
    sel = select_primary_outcomes(df, k_per_category=3)
    flat = []
    for cat, feats in sel.items():
        for rank, f in enumerate(feats, 1):
            flat.append({"corpus": corpus, "category": cat, "rank": rank, "feature": f})
    pd.DataFrame(flat).to_csv(out_dir / f"primary_outcomes_{corpus}.csv", index=False)
    with open(out_dir / f"primary_outcomes_{corpus}.json", "w") as fh:
        json.dump(sel, fh, indent=2)


def export_table4_latex(csv_path: pathlib.Path, out_path: pathlib.Path) -> None:
    df = pd.read_csv(csv_path)

    def stars(p):
        if pd.isna(p) or p >= 0.05:
            return ""
        if p < 0.001:
            return "^{***}"
        if p < 0.01:
            return "^{**}"
        if p < 0.05:
            return "^{*}"

    l1 = {"seame": "ZH", "masac": "HI"}
    lines = [
        "% Auto-generated by scripts/43_paper_tier_fixes.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{lllrrrrr}",
        "\\toprule",
        "\\textbf{Outcome} & \\textbf{Cat.} & \\textbf{Corpus} & \\textbf{Mono EN} & \\textbf{Mono L1} & \\textbf{DM} & \\textbf{Rep} & \\textbf{Speaker RE} \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        corp = r["corpus"]
        l1tag = l1[corp]
        lines.append(
            f"\\texttt{{{r['feature']}}} & {r['category']} & {corp.upper()} & "
            f"${r['beta_mono_en']:.1f}{stars(r['p_mono_en'])}$ & "
            f"${r['beta_mono_l1']:.1f}{stars(r['p_mono_l1'])}$ ({l1tag}) & "
            f"${r['beta_dm']:.1f}{stars(r['p_dm'])}$ & "
            f"${r['beta_rep']:.1f}{stars(r['p_rep'])}$ & "
            f"${r['speaker_re_var']:.3f}$ \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Primary mixed-effects coefficients (top-3 F0, duration, energy outcomes by between-condition separation variance; speaker random intercept; CS reference). "
        "RE = random-effect variance $\\sigma^2_s$. SEAME $n=106{,}841$; MaSaC $n=6{,}476$.}",
        "\\label{tab:mem-primary}",
        "\\end{table*}",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def sign_interpretation_doc(out_path: pathlib.Path) -> None:
    text = """# CS-as-reference sign convention (Table 4 / DM / Rep)

**Model:** `outcome ~ C(condition) + has_discourse_marker + has_repetition + (1|speaker)`  
**Reference level for `condition`:** `code_switched` (CS).  
**Reference for binary predictors:** `False` (no DM / no Rep).

## Condition coefficients (mono vs CS)

| Sign of mono EN / mono L1 coef | Interpretation |
|-------------------------------|----------------|
| **Negative** on F0 energy (e.g. F0msemax) | Monolingual utterances have **lower** pitch-excursion energy than CS |
| **Positive** on F0tiltmin | Monolingual utterances have **steeper** minimum tilt than CS |
| **Positive** on maxdurvoiced | Monolingual has **longer** max voiced duration than CS (check feature-specific direction) |

DB question (“shouldn’t DM signs be negative?”): **No** — DM and Rep are separate binary predictors with CS as the condition reference.  
A **positive** `has_discourse_marker[T.True]` on F0msemax means: among **CS utterances**, those **with** a discourse marker have **higher** F0msemax than CS utterances **without** a marker. It does **not** compare DMs to monolingual speech directly.

**Narrative:** Unmarked CS is the baseline; DM/Rep-marked CS turns are more prosodically animated (higher contour energy). Monolingual rows in the same table are negative vs CS, so marked CS is still generally closer to the animated end of the CS distribution, not above monolingual on all features.

## RQ3 interaction (condition × gender_match)

`outcome ~ C(condition) * C(gender_match) + ...` with CS reference and same-gender reference.

- `C(condition)[T.monolingual_EN]`: CS–mono EN gap when interlocutor is **same** gender.
- `C(condition)[T.monolingual_EN]:C(gender_match)[T.different]`: **change** in that gap under **mixed-gender** interlocutor.

If the interaction augments the mono coefficient in the direction of larger |gap|, mixed-gender turns show **stronger CS–monolingual separation** on that feature.
"""
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="results/paper/tier_fixes")
    args = p.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "seame": "results/functional_anchors/seame/seame_integrated_features.csv",
        "masac": "results/functional_anchors/masac/masac_integrated_features.csv",
    }
    table4_all = []
    for corpus, path in paths.items():
        df = pd.read_csv(path, low_memory=False)
        selection_manifest(df, corpus, out)
        t4 = export_table4(corpus, df, out)
        table4_all.append(t4)
        run_rq3(corpus, df, out, dyad_only=False)
        if corpus == "masac":
            dyad_path = pathlib.Path(
                "results/interlocutor_gender/masac_dyad/masac_integrated_features_dyad.csv"
            )
            if dyad_path.exists():
                df_dyad = pd.read_csv(dyad_path, low_memory=False)
                run_rq3(corpus, df_dyad, out, dyad_only=True)

    pd.concat(table4_all, ignore_index=True).to_csv(out / "table4_mem_primary_combined.csv", index=False)

    # Cross-corpus Cohen's d by category
    cat_rows = []
    for corp, pair in [
        ("seame", "CS_vs_EN"),
        ("seame", "CS_vs_ZH"),
        ("masac", "CS_vs_EN"),
        ("masac", "CS_vs_HI"),
    ]:
        fp = pathlib.Path(f"features/first_contrast_{corp}_{pair}.csv")
        if not fp.exists():
            continue
        s = category_cohen_summary(fp)
        s["corpus"] = corp
        s["comparison"] = pair
        cat_rows.append(s)
    if cat_rows:
        pd.concat(cat_rows, ignore_index=True).to_csv(out / "cohen_d_by_category.csv", index=False)

    # Speaker-level t-test on primary outcomes (clustering caveat)
    sp_rows = []
    for corpus, path in paths.items():
        df = pd.read_csv(path, low_memory=False)
        for feat in primary_outcome_list(df, k_per_category=3):
            t, p = speaker_level_ttest(df, feat)
            sp_rows.append(
                {
                    "corpus": corpus,
                    "feature": feat,
                    "speaker_level_t": t,
                    "speaker_level_p": p,
                }
            )
    pd.DataFrame(sp_rows).to_csv(out / "speaker_level_ttests_primary9.csv", index=False)

    sign_interpretation_doc(out / "SIGN_CONVENTION.md")

    # Cohen's d histogram (appendix figure)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
        specs = [
            ("seame", "CS_vs_EN", axes[0, 0]),
            ("seame", "CS_vs_ZH", axes[0, 1]),
            ("masac", "CS_vs_EN", axes[1, 0]),
            ("masac", "CS_vs_HI", axes[1, 1]),
        ]
        for corp, pair, ax in specs:
            fp = pathlib.Path(f"features/first_contrast_{corp}_{pair}.csv")
            if not fp.exists():
                continue
            d = pd.read_csv(fp)
            ax.hist(d["cohen_d"].dropna(), bins=30, color="#4C72B0", alpha=0.85)
            ax.axvline(0, color="k", lw=0.8)
            ax.set_title(f"{corp.upper()} {pair.replace('_', ' ')}")
            ax.set_ylabel("Count")
        for ax in axes[1]:
            ax.set_xlabel("Cohen's d (CS $-$ mono)")
        fig.suptitle("Effect-size distributions across 103 DisVoice features")
        fig.tight_layout()
        fig_dir = pathlib.Path("results/paper/figures")
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_dir / "fig_cohen_d_histogram.png", dpi=180)
        plt.close(fig)
    except Exception as e:
        print("Cohen d histogram skipped:", e)

    export_table4_latex(out / "table4_mem_primary_combined.csv", out / "TABLE4_LATEX.tex")

    # Corpus counts verification
    counts = []
    for corpus, path in paths.items():
        df = pd.read_csv(path, low_memory=False)
        sp = "speaker_id" if "speaker_id" in df.columns else "speaker"
        vc = df["condition"].value_counts()
        counts.append(
            {
                "corpus": corpus,
                "n_utterances": len(df),
                "n_speakers": df[sp].nunique(),
                "n_cs": int(vc.get("code_switched", 0)),
                "n_mono_en": int(vc.get("monolingual_EN", 0)),
                "n_mono_l1": int(
                    vc.get("monolingual_ZH", vc.get("monolingual_HI", 0))
                ),
                "dm_rate": float(df["has_discourse_marker"].mean()),
                "rep_rate": float(df["has_repetition"].mean()),
            }
        )
    pd.DataFrame(counts).to_csv(out / "corpus_counts_verified.csv", index=False)
    print(f"Wrote tier-fix outputs to {out}")


if __name__ == "__main__":
    main()
