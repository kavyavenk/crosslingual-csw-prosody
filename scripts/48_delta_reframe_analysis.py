"""
DB-thesis-style reframing of Section 5.2 / 5.3.

Instead of counting significant OLS/LME coefficients over CS-only utterances,
we report, for each contrast, the PROPORTION of the 103 DisVoice features that
differ significantly from *monolingual* speech (feature-level independent
t-tests, BH-FDR alpha=0.05), and compare that proportion between two CS
subgroups via a two-proportion z-test:

  richer vs sparser CS (M-index, I-index, n_switches)
  insertional vs alternational CS (strategy)
  hi/zh->en vs en->hi/zh (direction)
  with-DM vs without-DM CS ; with-Rep vs without-Rep
  (5.3) mixed- vs same-gender turns ; cross- vs same-age ; diff- vs same-nationality

For every subgroup G we compute p_G = (#features where G differs from pooled
monolingual speech at BH-FDR) / 103, then delta = p_high - p_low and a
two-proportion z-test on the counts.
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
    """The exact 103 DisVoice features used in the first-contrast tables."""
    fc = pd.read_csv("features/first_contrast_seame_CS_vs_EN.csv")
    return fc["feature"].tolist()


def feat_cols(df):
    canon = [f for f in canonical_103() if f in df.columns]
    return canon


def _cohend(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if sp == 0:
        return np.nan
    return (np.mean(a) - np.mean(b)) / sp


def prop_sig_vs_mono(cs_sub: pd.DataFrame, mono: pd.DataFrame, feats):
    """Return (#sig BH-FDR, n_tested, mean|Cohen d|) for cs_sub vs mono."""
    pvals = []
    ds = []
    for f in feats:
        a = cs_sub[f].dropna().values
        b = mono[f].dropna().values
        if len(a) < 3 or len(b) < 3:
            continue
        if np.nanstd(a) == 0 and np.nanstd(b) == 0:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        if np.isnan(p):
            continue
        pvals.append(p)
        d = _cohend(a, b)
        if not np.isnan(d):
            ds.append(abs(d))
    if not pvals:
        return 0, 0, float("nan")
    rej, _, _, _ = multipletests(pvals, alpha=ALPHA, method="fdr_bh")
    return int(rej.sum()), len(pvals), float(np.mean(ds)) if ds else float("nan")


def contrast(df, feats, mono, mask_high, mask_low, label_high, label_low, name):
    kh, nh, dh = prop_sig_vs_mono(df[mask_high], mono, feats)
    kl, nl, dl = prop_sig_vs_mono(df[mask_low], mono, feats)
    ph = kh / nh if nh else 0.0
    pl = kl / nl if nl else 0.0
    try:
        z, pz = proportions_ztest([kh, kl], [nh, nl])
    except Exception:
        z, pz = np.nan, np.nan
    delta = ph - pl
    print(
        f"  {name:26s} {label_high:>13s} {100*ph:5.1f}% ({kh}/{nh}) d={dh:.2f} | "
        f"{label_low:>13s} {100*pl:5.1f}% ({kl}/{nl}) d={dl:.2f} | "
        f"Δ%={100*delta:+5.1f}pp (z p={pz:.2g})  Δ|d|={dh-dl:+.2f}"
    )
    return dict(name=name, p_high=ph, p_low=pl, delta=delta, k_high=kh, n_high=nh,
                k_low=kl, n_low=nl, z=z, pz=pz, d_high=dh, d_low=dl, delta_d=dh - dl)


def run_corpus(path, corpus, l1_name):
    print(f"\n================= {corpus.upper()} =================")
    df = pd.read_csv(path, low_memory=False)
    feats = feat_cols(df)
    print(f"n={len(df)}  features tested up to {len(feats)}")
    cs = df["condition"] == "code_switched"
    mono = df[df["condition"].isin(["monolingual_EN", f"monolingual_{l1_name}"])]
    csdf = df[cs].copy()

    rows = []
    # ---- richness / quantity: median split within CS ----
    for prop in ["m_index", "i_index", "n_switches"]:
        if prop not in csdf:
            continue
        med = csdf[prop].median()
        hi = csdf[prop] > med
        lo = csdf[prop] <= med
        rows.append(contrast(csdf, feats, mono, hi, lo, "richer", "sparser",
                             f"{prop} (median split)"))
    # ---- direction ----
    if "is_l1_to_l2" in csdf:
        # Empirically, is_l1_to_l2 == 1 matches the "toward en" share in Table 1
        # (SEAME 61.8%, MaSaC 58.1%); the complement is toward the L1.
        toward_en = csdf["is_l1_to_l2"] == 1
        toward_l1 = csdf["is_l1_to_l2"] == 0
        rows.append(contrast(csdf, feats, mono, toward_en, toward_l1,
                             "->en", f"->{l1_name.lower()}", "direction"))
    # ---- strategy ----
    if "cs_strategy" in csdf:
        s = csdf["cs_strategy"].astype(str).str.lower()
        ins = s.str.startswith("ins") | (s == "i")
        alt = s.str.startswith("alt") | (s == "a")
        rows.append(contrast(csdf, feats, mono, ins, alt,
                             "insertional", "alternational", "strategy"))
    # ---- discourse markers / repetition ----
    for prop, hi_lab, lo_lab in [("has_discourse_marker", "with-DM", "no-DM"),
                                 ("has_repetition", "with-Rep", "no-Rep")]:
        if prop in csdf:
            hi = csdf[prop].astype(bool)
            lo = ~csdf[prop].astype(bool)
            rows.append(contrast(csdf, feats, mono, hi, lo, hi_lab, lo_lab, prop))
    return pd.DataFrame(rows).assign(corpus=corpus)


if __name__ == "__main__":
    out = []
    out.append(run_corpus("results/functional_anchors/seame/seame_integrated_features.csv",
                          "seame", "ZH"))
    out.append(run_corpus("results/functional_anchors/masac/masac_integrated_features.csv",
                          "masac", "HI"))
    res = pd.concat(out, ignore_index=True)
    res.to_csv("results/paper/delta_reframe_5_2.csv", index=False)
    print("\nSaved -> results/paper/delta_reframe_5_2.csv")
