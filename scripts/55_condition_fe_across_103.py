"""
For all 103 DisVoice features per corpus, refit the primary mixed-effects model
(Eq. mem-primary, speaker random intercept) and test whether the CONDITION fixed
effect (CSW vs. mono-EN vs. mono-L1) remains jointly significant AFTER
controlling for the speaker random intercept.

Reports the proportion of features with a significant condition fixed effect
(joint Wald test on the two condition dummies, BH-FDR corrected across the 103
features), and the proportion where BOTH the speaker random intercept (from
50_speaker_re_across_103.py) and the condition fixed effect are significant.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

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
    except Exception:
        return None
    # joint Wald test on all C(condition) fixed-effect terms
    names = list(mix.params.index)
    cond_pos = [i for i, n in enumerate(names) if n.startswith("C(condition)")]
    if not cond_pos:
        return None
    R = np.zeros((len(cond_pos), len(names)))
    for j, pos in enumerate(cond_pos):
        R[j, pos] = 1.0
    try:
        wt = mix.wald_test(R, scalar=True)
        p_cond = float(np.asarray(wt.pvalue).ravel()[0])
    except Exception:
        return None
    return dict(feature=feat, p_condition=p_cond, n=len(m))


def run(path, corpus):
    df = pd.read_csv(path, low_memory=False)
    feats = [f for f in canonical_103() if f in df.columns]
    out_path = f"results/paper/condition_fe_{corpus}.csv"
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
            pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  [{corpus}] {i}/{len(feats)} ({feat})", flush=True)
    res = pd.DataFrame(rows)
    res["sig_condition_fdr"] = multipletests(res["p_condition"], alpha=ALPHA, method="fdr_bh")[0]
    res.to_csv(out_path, index=False)

    # join with speaker RE significance
    re_path = f"results/paper/speaker_re_{corpus}.csv"
    both = None
    if os.path.exists(re_path):
        re = pd.read_csv(re_path)[["feature", "sig"]].rename(columns={"sig": "sig_re"})
        j = res.merge(re, on="feature", how="inner")
        both = int((j["sig_condition_fdr"] & j["sig_re"]).sum())
        n_both = len(j)
    n = len(res)
    nsig = int(res["sig_condition_fdr"].sum())
    print(f"\n{corpus.upper()}: condition FE significant (BH-FDR) for "
          f"{nsig}/{n} = {100*nsig/n:.0f}% of features (after controlling for speaker)")
    if both is not None:
        print(f"{corpus.upper()}: BOTH speaker RE and condition FE significant for "
              f"{both}/{n_both} = {100*both/n_both:.0f}%")
    return res


if __name__ == "__main__":
    run("results/functional_anchors/masac/masac_integrated_features.csv", "masac")
    run("results/functional_anchors/seame/seame_integrated_features.csv", "seame")
    print("\nDONE")
