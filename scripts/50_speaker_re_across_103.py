"""
Fit the primary mixed-effects model (Eq. mem-primary) for ALL 103 DisVoice
features per corpus and report the proportion whose speaker random-intercept
variance is significant.

Significance is assessed with a likelihood-ratio test comparing the mixed model
(with a by-speaker random intercept) to the corresponding OLS model (no random
effect), df=1, using the 0.5*chi2_1 boundary-corrected p-value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2

import paper_feature_selection as pfs

ALPHA = 0.05


def canonical_103():
    return pd.read_csv("features/first_contrast_seame_CS_vs_EN.csv")["feature"].tolist()


def lhs(col):
    if col.isidentifier():
        return col
    return 'Q("%s")' % col.replace("\\", "\\\\").replace('"', '\\"')


def fit_feature(df, feat, group="speaker_id"):
    cols = [feat, "condition", "has_discourse_marker", "has_repetition", group]
    m = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(m) < 100 or m[feat].std() == 0:
        return None
    f = f"{lhs(feat)} ~ C(condition) + has_discourse_marker + has_repetition"
    try:
        mix = smf.mixedlm(f, m, groups=m[group]).fit(method=["lbfgs", "powell"])
        ols = smf.ols(f, m).fit()
    except Exception:
        return None
    re_var = float(mix.cov_re.iloc[0, 0]) if hasattr(mix, "cov_re") else np.nan
    lr = 2 * (mix.llf - ols.llf)
    if lr < 0:
        lr = 0.0
    # boundary-corrected mixture: 0.5*chi2_0 + 0.5*chi2_1
    p = 0.5 * chi2.sf(lr, 1)
    return dict(feature=feat, re_var=re_var, lr=lr, p=p, sig=bool(p < ALPHA))


def run(path, corpus, l1):
    import os
    df = pd.read_csv(path, low_memory=False)
    feats = [f for f in canonical_103() if f in df.columns]
    out_path = f"results/paper/speaker_re_{corpus}.csv"
    done = {}
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        done = {r["feature"]: r.to_dict() for _, r in prev.iterrows()}
    rows = list(done.values())
    for i, feat in enumerate(feats, 1):
        if feat in done:
            continue
        r = fit_feature(df, feat)
        if r:
            rows.append(r)
            pd.DataFrame(rows).to_csv(out_path, index=False)  # incremental save
        print(f"  [{corpus}] {i}/{len(feats)} ({feat})", flush=True)
    res = pd.DataFrame(rows)
    n = len(res)
    nsig = int(res["sig"].sum())
    print(f"\n{corpus.upper()}: speaker RE significant for {nsig}/{n} features "
          f"= {100*nsig/n:.0f}%", flush=True)
    res.to_csv(f"results/paper/speaker_re_{corpus}.csv", index=False)
    return nsig, n


if __name__ == "__main__":
    run("results/functional_anchors/masac/masac_integrated_features.csv", "masac", "HI")
    run("results/functional_anchors/seame/seame_integrated_features.csv", "seame", "ZH")
    print("\nDONE")
