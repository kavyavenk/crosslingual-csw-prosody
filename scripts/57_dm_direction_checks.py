"""
Rechecks for the 5.2 discourse-marker x direction KV comments.

(A) DM x direction delta vs monolingual (KV: "how does DM change the delta w.r.t.
    monolingual within a given direction?"). Within each switch direction, compare
    the proportion of the 103 features that differ (BH-FDR, Welch t) from pooled
    monolingual speech for DM-present vs DM-absent CSW.

(B) SEAME direction x DM interaction (KV: "where are these betas from?"). CS-only
    OLS on the ten highest-variance F0 outcomes with the appendix direction spec;
    report the is_l1_to_l2:has_discourse_marker interaction beta + BH-FDR count.

Also prints the corrected MaSaC DM x direction chi-square / Cramer's V.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

import paper_feature_selection as pfs

ALPHA = 0.05


def canonical_103():
    return pd.read_csv("features/first_contrast_seame_CS_vs_EN.csv")["feature"].tolist()


def cohend(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp if sp else np.nan


def prop_sig(sub, mono, feats):
    ps, ds = [], []
    for f in feats:
        a = sub[f].dropna().values
        b = mono[f].dropna().values
        if len(a) < 3 or len(b) < 3:
            continue
        _, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        if not np.isnan(p):
            ps.append(p)
            d = cohend(a, b)
            if not np.isnan(d):
                ds.append(abs(d))
    if not ps:
        return 0, 0, float("nan")
    rej = multipletests(ps, alpha=ALPHA, method="fdr_bh")[0]
    return int(rej.sum()), len(ps), float(np.mean(ds)) if ds else float("nan")


def lhs(c):
    return c if c.isidentifier() else 'Q("%s")' % c


def partA(df, corpus, l1):
    print(f"\n=== (A) {corpus.upper()}: DM x direction delta vs monolingual ===")
    feats = [f for f in canonical_103() if f in df.columns]
    mono = df[df["condition"].isin(["monolingual_EN", f"monolingual_{l1}"])]
    cs = df[df["condition"] == "code_switched"].copy()
    cs["dm"] = cs["has_discourse_marker"].astype(bool)
    dir_names = {1: f"{l1.lower()}->en (toward en)", 0: f"en->{l1.lower()}"}
    for dv in [1, 0]:
        sub = cs[cs["is_l1_to_l2"] == dv]
        dm1 = sub[sub["dm"]]
        dm0 = sub[~sub["dm"]]
        k1, n1, d1 = prop_sig(dm1, mono, feats)
        k0, n0, d0 = prop_sig(dm0, mono, feats)
        p1 = k1 / n1 if n1 else 0
        p0 = k0 / n0 if n0 else 0
        try:
            z, pz = proportions_ztest([k1, k0], [n1, n0])
        except Exception:
            z, pz = np.nan, np.nan
        print(f"  {dir_names[dv]}: DM+ {100*p1:5.1f}% ({k1}/{n1}) d={d1:.2f} nutt={len(dm1)} | "
              f"DM- {100*p0:5.1f}% ({k0}/{n0}) d={d0:.2f} nutt={len(dm0)} | "
              f"delta={100*(p1-p0):+5.1f}pp (p={pz:.2g}) delta|d|={d1-d0:+.2f}")


def partB_seame(df):
    print("\n=== (B) SEAME: direction x DM interaction (CS-only OLS, top-10 F0) ===")
    feats = [f for f in canonical_103() if f in df.columns]
    cs = df[df["condition"] == "code_switched"].copy()
    f0 = [c for c in feats if pfs.disvoice_category(c) == "f0"]
    top = cs[f0].var().sort_values(ascending=False).index[:10].tolist()
    rows = []
    for f in top:
        cols = [f, "is_l1_to_l2", "has_discourse_marker", "has_repetition",
                "gender", "age_bucket", "nationality"]
        m = cs[cols].replace([np.inf, -np.inf], np.nan).dropna()
        if m[f].std() == 0:
            continue
        formula = (f"{lhs(f)} ~ is_l1_to_l2 * has_discourse_marker + has_repetition "
                   f"+ C(gender) + C(age_bucket) + C(nationality)")
        r = smf.ols(formula, m).fit()
        term = [t for t in r.params.index if ":" in t and "is_l1_to_l2" in t]
        if not term:
            continue
        t = term[0]
        rows.append((f, float(r.params[t]), float(r.pvalues[t])))
    res = pd.DataFrame(rows, columns=["feature", "beta_int", "p"])
    res["sig"] = multipletests(res["p"], alpha=ALPHA, method="fdr_bh")[0]
    print(res.round(3).to_string())
    print(f"  significant direction x DM interactions: {int(res['sig'].sum())}/{len(res)}")


def masac_chi(df):
    cs = df[df["condition"] == "code_switched"]
    ct = pd.crosstab(cs["has_discourse_marker"].astype(bool), (cs["is_l1_to_l2"] == 1))
    chi2y, py, _, _ = chi2_contingency(ct.values)
    chi2u, pu, _, _ = chi2_contingency(ct.values, correction=False)
    n = ct.values.sum()
    V = np.sqrt(chi2u / n)
    print("\n=== MaSaC DM x direction association ===")
    print(f"  chi2(Yates)={chi2y:.1f} (uncorrected={chi2u:.1f}), p={py:.1e}, Cramer's V={V:.3f}, n={n}")


if __name__ == "__main__":
    masac = pd.read_csv("results/functional_anchors/masac/masac_integrated_features.csv", low_memory=False)
    seame = pd.read_csv("results/functional_anchors/seame/seame_integrated_features.csv", low_memory=False)
    masac_chi(masac)
    partA(masac, "masac", "HI")
    partB_seame(seame)
    print("\nDONE")
