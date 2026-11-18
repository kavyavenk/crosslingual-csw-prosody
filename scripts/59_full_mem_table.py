"""
Full auto-ranked mixed-effects table (Appendix app:mem-full).

Fits the primary LME (Eq. mem-primary; CSW reference; speaker random intercept)
for ALL 103 DisVoice features per corpus and dumps the monolingual-EN and
monolingual-L1 fixed-effect coefficients + p-values, ranked by between-condition
separation variance (same rule as the main-text primary outcomes). Resumable.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import paper_feature_selection as pfs

ALPHA = 0.05


def canonical_103():
    return pd.read_csv("features/first_contrast_seame_CS_vs_EN.csv")["feature"].tolist()


def lhs(c):
    return c if c.isidentifier() else 'Q("%s")' % c


def fit(df, feat, l1, group="speaker_id"):
    cols = [feat, "condition", "has_discourse_marker", "has_repetition", group]
    m = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(m) < 100 or m[feat].std() == 0:
        return None
    f = f"{lhs(feat)} ~ C(condition) + has_discourse_marker + has_repetition"
    try:
        mix = smf.mixedlm(f, m, groups=m[group]).fit(method=["lbfgs", "powell"])
    except Exception:
        return None
    p = mix.params
    pv = mix.pvalues
    en = [i for i in p.index if "monolingual_EN" in i]
    l1t = [i for i in p.index if f"monolingual_{l1}" in i]
    sep = pfs.condition_separation_variance(df, feat)
    return dict(
        feature=feat,
        category=pfs.disvoice_category(feat),
        beta_en=float(p[en[0]]) if en else np.nan,
        p_en=float(pv[en[0]]) if en else np.nan,
        beta_l1=float(p[l1t[0]]) if l1t else np.nan,
        p_l1=float(pv[l1t[0]]) if l1t else np.nan,
        sep_var=sep,
    )


def run(path, corpus, l1):
    df = pd.read_csv(path, low_memory=False)
    feats = [f for f in canonical_103() if f in df.columns]
    out_path = f"results/paper/mem_full_{corpus}.csv"
    done = {}
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        done = {r["feature"]: r.to_dict() for _, r in prev.iterrows()}
    rows = list(done.values())
    for i, feat in enumerate(feats, 1):
        if feat in done:
            continue
        r = fit(df, feat, l1)
        if r:
            rows.append(r)
            pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  [{corpus}] {i}/{len(feats)} {feat}", flush=True)
    res = pd.DataFrame(rows).sort_values("sep_var", ascending=False)
    res.to_csv(out_path, index=False)
    print(f"{corpus.upper()}: wrote {out_path} ({len(res)} features)")


if __name__ == "__main__":
    run("results/functional_anchors/masac/masac_integrated_features.csv", "masac", "HI")
    run("results/functional_anchors/seame/seame_integrated_features.csv", "seame", "ZH")
    print("\nDONE")
