"""
[[Kavya-1, complement]] Multivariate divergence model (directly answers reviewer
hoSr W3's suggestion of "multivariate or dimension-reduced analyses of the overall
prosodic profile", avoiding the 103-way multiplicity of the per-feature tests).

For each utterance we build a single scalar PROSODIC DIVERGENCE score = the mean
absolute z-distance of its 103 DisVoice features from the pooled monolingual
prototype (feature means/SDs computed on monolingual EN+L1 speech; z clipped at
+/-8 to bound outliers). Larger = more prosodically unlike monolingual speech.

We then fit a by-speaker hierarchical model over CSW utterances:
    divergence ~ has_dm + has_rep + n_switches_z + is_insertional + (1|speaker)
A positive, significant coefficient means that property AMPLIFIES overall
divergence from monolingual speech -- the amplification claim tested directly on
the whole prosodic profile at once.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.simplefilter("ignore")

FILES = {
    "seame": ("results/functional_anchors/seame/seame_integrated_features.csv", "ZH"),
    "masac": ("results/functional_anchors/masac/masac_integrated_features.csv", "HI"),
}


def canonical_103():
    return pd.read_csv("features/first_contrast_seame_CS_vs_EN.csv")["feature"].tolist()


def divergence_score(df, feats, mono):
    mu = mono[feats].mean()
    sd = mono[feats].std(ddof=1).replace(0, np.nan)
    z = (df[feats] - mu) / sd
    z = z.clip(-8, 8)
    return z.abs().mean(axis=1)


def run(corpus, path, l1):
    df = pd.read_csv(path, low_memory=False)
    feats = [f for f in canonical_103() if f in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan)
    mono = df[df["condition"].isin(["monolingual_EN", f"monolingual_{l1}"])]
    df["divergence"] = divergence_score(df, feats, mono)

    print(f"\n================= {corpus.upper()} =================")
    cs = df[df["condition"] == "code_switched"]
    print(f"mean divergence: monolingual={df.loc[df.condition!='code_switched','divergence'].mean():.3f}"
          f"  CSW={cs['divergence'].mean():.3f}")

    cs = cs.copy()
    cs["has_dm"] = cs["has_discourse_marker"].astype(int)
    cs["has_rep"] = cs["has_repetition"].astype(int)
    n = pd.to_numeric(cs["n_switches"], errors="coerce")
    cs["n_switches_z"] = (n - n.mean()) / n.std(ddof=1)
    s = cs["cs_strategy"].astype(str).str.lower()
    cs["is_insertional"] = np.where(s.str.startswith("ins"), 1,
                                    np.where(s.str.startswith("alt"), 0, np.nan))

    m = cs[["divergence", "has_dm", "has_rep", "n_switches_z",
            "is_insertional", "speaker_id"]].dropna().reset_index(drop=True)
    r = smf.mixedlm(
        "divergence ~ has_dm + has_rep + n_switches_z + is_insertional",
        m, groups=m["speaker_id"],
    ).fit(method=["lbfgs", "powell"])
    print(f"CSW-only divergence model (n={len(m)}, {m['speaker_id'].nunique()} speakers):")
    for term in ["has_dm", "has_rep", "n_switches_z", "is_insertional"]:
        b, p = r.params[term], r.pvalues[term]
        arrow = "amplifies" if b > 0 else "reduces"
        sig = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."
        print(f"  {term:14s} beta={b:+.4f} p={p:.2g} {sig:4s} -> {arrow} divergence")
    return r


if __name__ == "__main__":
    for corpus, (path, l1) in FILES.items():
        run(corpus, path, l1)
    print("\nDONE")
