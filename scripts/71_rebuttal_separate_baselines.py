"""
[[Kavya-2]] Reviewer hoSr W4 / comment [5]: the Section 5.2 subgroup + discourse
contrasts pool monolingual English with monolingual Mandarin/Hindi. Re-run them
against each monolingual baseline SEPARATELY (mono-EN only, mono-L1 only) and
check whether the DIRECTION and SIGNIFICANCE of the key effects -- especially
discourse-marker amplification and nswitches richness -- hold against both.

For each binarized CSW property we compute, against a given monolingual baseline,
   p_hi = share of 103 features on which the "high" subgroup differs (BH-FDR Welch)
   p_lo = same for the "low" subgroup
   delta = p_hi - p_lo, with a two-proportion z-test.
We report delta + p for baseline in {pooled, EN, L1}.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

ALPHA = 0.05


def canonical_103():
    return pd.read_csv("features/first_contrast_seame_CS_vs_EN.csv")["feature"].tolist()


def prop_sig(sub, mono, feats):
    ps = []
    for f in feats:
        a = sub[f].dropna().values
        b = mono[f].dropna().values
        if len(a) < 3 or len(b) < 3:
            continue
        _, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        if not np.isnan(p):
            ps.append(p)
    if not ps:
        return 0, 0
    rej = multipletests(ps, alpha=ALPHA, method="fdr_bh")[0]
    return int(rej.sum()), len(ps)


def contrast(cs, feats, mono, hi_mask, lo_mask):
    kh, nh = prop_sig(cs[hi_mask], mono, feats)
    kl, nl = prop_sig(cs[lo_mask], mono, feats)
    ph = kh / nh if nh else 0.0
    pl = kl / nl if nl else 0.0
    try:
        z, pz = proportions_ztest([kh, kl], [nh, nl])
    except Exception:
        z, pz = np.nan, np.nan
    return dict(p_hi=ph, p_lo=pl, delta=ph - pl, pz=pz, kh=kh, nh=nh, kl=kl, nl=nl)


def run(corpus, path, l1):
    df = pd.read_csv(path, low_memory=False)
    feats = [f for f in canonical_103() if f in df.columns]
    cs = df[df["condition"] == "code_switched"].copy()
    cs["n_switches"] = pd.to_numeric(cs["n_switches"], errors="coerce")

    baselines = {
        "pooled": df[df["condition"].isin(["monolingual_EN", f"monolingual_{l1}"])],
        "EN": df[df["condition"] == "monolingual_EN"],
        f"{l1}": df[df["condition"] == f"monolingual_{l1}"],
    }

    s = cs["cs_strategy"].astype(str).str.lower()
    med = cs["n_switches"].median()
    props = {
        "nswitches (hi vs lo)": (cs["n_switches"] > med, cs["n_switches"] <= med),
        "discourse marker (+/-)": (cs["has_discourse_marker"].astype(bool),
                                   ~cs["has_discourse_marker"].astype(bool)),
        "repetition (+/-)": (cs["has_repetition"].astype(bool),
                             ~cs["has_repetition"].astype(bool)),
        "strategy (ins vs alt)": (s.str.startswith("ins"), s.str.startswith("alt")),
        "direction (->en vs ->L1)": (cs["is_l1_to_l2"] == 1, cs["is_l1_to_l2"] == 0),
    }

    print(f"\n================= {corpus.upper()} =================")
    rows = []
    for pname, (hi, lo) in props.items():
        line = f"  {pname:26s}"
        for bname, mono in baselines.items():
            r = contrast(cs, feats, mono, hi, lo)
            sig = "*" if (r["pz"] is not None and r["pz"] < 0.05) else " "
            line += f" | {bname}: {100*r['delta']:+6.1f}pp{sig}(p={r['pz']:.1g})"
            rows.append(dict(corpus=corpus, property=pname, baseline=bname, **r))
        print(line)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = []
    out.append(run("seame", "results/functional_anchors/seame/seame_integrated_features.csv", "ZH"))
    out.append(run("masac", "results/functional_anchors/masac/masac_integrated_features.csv", "HI"))
    res = pd.concat(out, ignore_index=True)
    res.to_csv("results/paper/validation/kavya2_separate_baselines.csv", index=False)
    print("\nSaved -> results/paper/validation/kavya2_separate_baselines.csv")
