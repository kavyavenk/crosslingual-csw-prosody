"""
Quick appendix/KV computations:

(A) KV reframe: [CSW with DM AND Rep] vs monolingual compared to
    [CSW with NEITHER DM nor Rep] vs monolingual -- proportion of 103 features
    differing (BH-FDR, Welch t) + mean|d| + two-proportion z-test.

(B) app:effect-sizes: mean |Cohen's d| per prosodic category per comparison,
    from the first_contrast_* files.

(C) app:functional-anchors: per-category (F0/duration/energy) proportion of
    features on which CSW differs from monolingual, stratified by discourse-marker
    presence and by repetition presence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
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


def partA(dfs):
    print("\n=== (A) [CSW w/ DM AND Rep] vs mono  vs  [CSW w/ NEITHER] vs mono ===")
    for corpus, df, l1 in dfs:
        feats = [f for f in canonical_103() if f in df.columns]
        mono = df[df["condition"].isin(["monolingual_EN", f"monolingual_{l1}"])]
        cs = df[df["condition"] == "code_switched"].copy()
        dm = cs["has_discourse_marker"].astype(bool)
        rp = cs["has_repetition"].astype(bool)
        both = cs[dm & rp]
        neither = cs[~dm & ~rp]
        kb, nb, db_ = prop_sig(both, mono, feats)
        kn, nn, dn = prop_sig(neither, mono, feats)
        pb, pn = kb / nb if nb else 0, kn / nn if nn else 0
        z, pz = proportions_ztest([kb, kn], [nb, nn])
        print(f"  {corpus.upper()}: DM&Rep {100*pb:5.1f}% ({kb}/{nb}) d={db_:.2f} n={len(both)} | "
              f"NEITHER {100*pn:5.1f}% ({kn}/{nn}) d={dn:.2f} n={len(neither)} | "
              f"delta={100*(pb-pn):+5.1f}pp (p={pz:.2g}) delta|d|={db_-dn:+.2f}")


def partB():
    print("\n=== (B) mean|Cohen's d| per category per comparison ===")
    files = [("SEAME", "CS vs EN", "features/first_contrast_seame_CS_vs_EN.csv"),
             ("SEAME", "CS vs ZH", "features/first_contrast_seame_CS_vs_ZH.csv"),
             ("MaSaC", "CS vs EN", "features/first_contrast_masac_CS_vs_EN.csv"),
             ("MaSaC", "CS vs HI", "features/first_contrast_masac_CS_vs_HI.csv")]
    for corpus, comp, path in files:
        d = pd.read_csv(path)
        d["cat"] = d["feature"].map(pfs.disvoice_category)
        d["absd"] = d["cohen_d"].abs()
        by = d.groupby("cat")["absd"].mean()
        overall = d["absd"].mean()
        print(f"  {corpus} {comp}: overall|d|={overall:.2f} | "
              f"F0={by.get('f0', float('nan')):.2f} dur={by.get('duration', float('nan')):.2f} "
              f"energy={by.get('energy', float('nan')):.2f}")


def partC(dfs):
    print("\n=== (C) per-category CSW-vs-mono share, stratified by DM / Rep ===")
    for corpus, df, l1 in dfs:
        feats = [f for f in canonical_103() if f in df.columns]
        mono = df[df["condition"].isin(["monolingual_EN", f"monolingual_{l1}"])]
        cs = df[df["condition"] == "code_switched"].copy()
        cats = {c: [f for f in feats if pfs.disvoice_category(f) == c] for c in ("f0", "duration", "energy")}
        for strat, col in [("DM", "has_discourse_marker"), ("Rep", "has_repetition")]:
            pres = cs[cs[col].astype(bool)]
            absent = cs[~cs[col].astype(bool)]
            print(f"  {corpus.upper()} [{strat}]:")
            for c, fl in cats.items():
                kp, npp, _ = prop_sig(pres, mono, fl)
                ka, na, _ = prop_sig(absent, mono, fl)
                pp = 100 * kp / npp if npp else 0
                pa = 100 * ka / na if na else 0
                print(f"      {c:8s}: present {pp:5.1f}% ({kp}/{npp}) | absent {pa:5.1f}% ({ka}/{na})")


if __name__ == "__main__":
    seame = pd.read_csv("results/functional_anchors/seame/seame_integrated_features.csv", low_memory=False)
    masac = pd.read_csv("results/functional_anchors/masac/masac_integrated_features.csv", low_memory=False)
    dfs = [("seame", seame, "ZH"), ("masac", masac, "HI")]
    partA(dfs)
    partB()
    partC(dfs)
    print("\nDONE")
