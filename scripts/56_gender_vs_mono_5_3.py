"""
Section 5.3 reframe (speaker gender), mirroring the 5.2 delta framing.

Part A (proportion-of-significant-features deltas):
  For each corpus, compare FEMALE code-switched vs pooled monolingual and MALE
  code-switched vs pooled monolingual: the proportion of the 103 DisVoice
  features that differ significantly (BH-FDR, Welch t) from monolingual speech,
  with mean |Cohen's d|. Female-vs-male delta tested with a two-proportion z-test.

Part B (OLS):
  CS-only OLS on the top-variance F0 outcomes (non-tilt) with speaker gender as
  predictor (female = reference); report how many features show a significant
  gender effect (BH-FDR) and example betas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
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
    pvals, ds = [], []
    for f in feats:
        a = sub[f].dropna().values
        b = mono[f].dropna().values
        if len(a) < 3 or len(b) < 3:
            continue
        _, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        if not np.isnan(p):
            pvals.append(p)
            d = cohend(a, b)
            if not np.isnan(d):
                ds.append(abs(d))
    if not pvals:
        return 0, 0, float("nan")
    rej = multipletests(pvals, alpha=ALPHA, method="fdr_bh")[0]
    return int(rej.sum()), len(pvals), float(np.mean(ds)) if ds else float("nan")


def lhs(c):
    return c if c.isidentifier() else 'Q("%s")' % c


def run(df, corpus, l1):
    print(f"\n================= {corpus.upper()} =================")
    df = df.copy()
    df["gender"] = df["gender"].astype(str).str.lower()
    feats = [f for f in canonical_103() if f in df.columns]
    mono = df[df["condition"].isin(["monolingual_EN", f"monolingual_{l1}"])]
    cs = df[df["condition"] == "code_switched"]
    fem = cs[cs["gender"] == "female"]
    mal = cs[cs["gender"] == "male"]

    kf, nf, df_d = prop_sig(fem, mono, feats)
    km, nm, dm_d = prop_sig(mal, mono, feats)
    pf, pm = kf / nf if nf else 0, km / nm if nm else 0
    z, pz = proportions_ztest([kf, km], [nf, nm])
    print("Part A -- % of 103 features differing from monolingual (BH-FDR):")
    print(f"  FEMALE CS vs mono: {100*pf:5.1f}% ({kf}/{nf})  mean|d|={df_d:.2f}  n={len(fem)}")
    print(f"  MALE   CS vs mono: {100*pm:5.1f}% ({km}/{nm})  mean|d|={dm_d:.2f}  n={len(mal)}")
    print(f"  delta(F-M) = {100*(pf-pm):+.1f}pp  (two-prop z p={pz:.2g})  delta|d|={df_d-dm_d:+.2f}")

    # Part B: CS-only OLS, gender effect on top-variance non-tilt F0 features
    f0 = [c for c in feats if pfs.disvoice_category(c) == "f0" and "tilt" not in c.lower()]
    top = cs[f0].var().sort_values(ascending=False).index[:10].tolist()
    csg = cs[cs["gender"].isin(["female", "male"])].copy()
    rows = []
    for f in top:
        m = csg[[f, "gender"]].replace([np.inf, -np.inf], np.nan).dropna()
        if m[f].std() == 0 or m["gender"].nunique() < 2:
            continue
        r = smf.ols(f'{lhs(f)} ~ C(gender, Treatment(reference="female"))', m).fit()
        term = [t for t in r.params.index if t.startswith("C(gender")][0]
        rows.append((f, float(r.params[term]), float(r.pvalues[term])))
    res = pd.DataFrame(rows, columns=["feature", "beta_male", "p"])
    res["sig"] = multipletests(res["p"], alpha=ALPHA, method="fdr_bh")[0]
    nsig = int(res["sig"].sum())
    print(f"Part B -- CS-only OLS gender effect (male vs female ref) on top-10 non-tilt F0:")
    print(f"  significant for {nsig}/{len(res)} features (BH-FDR)")
    ex = res[res["sig"]].reindex(res["p"].sort_values().index).head(3)
    for _, r in ex.iterrows():
        print(f"    {r['feature']}: beta_male={r['beta_male']:+.1f} (p={r['p']:.1e})")
    return dict(corpus=corpus, p_female=pf, p_male=pm, k_female=kf, n_female=len(fem),
                k_male=km, n_male=len(mal), delta=pf - pm, z=z, pz=pz,
                d_female=df_d, d_male=dm_d, ols_sig=nsig, ols_n=len(res))


if __name__ == "__main__":
    seame = pd.read_csv("results/functional_anchors/seame/seame_integrated_features.csv", low_memory=False)
    masac = pd.read_csv("results/interlocutor_gender/masac_dyad/masac_integrated_features_dyad.csv", low_memory=False)
    out = [run(seame, "seame", "ZH"), run(masac, "masac", "HI")]
    pd.DataFrame(out).to_csv("results/paper/gender_vs_mono_5_3.csv", index=False)
    print("\nSaved -> results/paper/gender_vs_mono_5_3.csv")
