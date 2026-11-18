"""
Real corpus examples (AHu9 typo): surface concrete SEAME/MaSaC utterances to
replace schematic examples -- illustrating each monolingual/CSW category and the
discourse-marker / repetition strata.

Prints ranked candidates per category; we then hand-pick 2-3 short, clean ones.
"""
from __future__ import annotations

import pandas as pd

pd.set_option("display.max_colwidth", 120)

FILES = {
    "seame": ("results/functional_anchors/seame/seame_integrated_features.csv", "ZH"),
    "masac": ("results/functional_anchors/masac/masac_integrated_features.csv", "HI"),
}


def wc(s):
    return len(str(s).split())


def show(df, title, n=6):
    cols = [c for c in ["utt_id", "text", "n_switches", "cs_strategy",
                        "discourse_marker_type", "n_repetitions"] if c in df.columns]
    print(f"\n--- {title} (n={len(df)}) ---")
    if len(df):
        print(df[cols].head(n).to_string(index=False))


def run(corpus, path, l1):
    df = pd.read_csv(path, low_memory=False)
    df["_wc"] = df["text"].map(wc)
    # keep readable, short-ish utterances
    ok = df[(df["_wc"] >= 3) & (df["_wc"] <= 14)].copy()
    csw = ok[ok["condition"] == "code_switched"]
    print(f"\n================= {corpus.upper()} =================")

    show(ok[ok["condition"] == "monolingual_EN"].sort_values("_wc"), "Monolingual English")
    show(ok[ok["condition"] == f"monolingual_{l1}"].sort_values("_wc"),
         f"Monolingual {l1}")

    ins = csw[csw["cs_strategy"].astype(str).str.lower().str.startswith("ins")]
    alt = csw[csw["cs_strategy"].astype(str).str.lower().str.startswith("alt")]
    show(ins[ins["n_switches"] == 1].sort_values("_wc"), "CSW insertional (single switch)")
    show(alt.sort_values("_wc"), "CSW alternational")

    dm = csw[csw["has_discourse_marker"].astype(bool)]
    show(dm.sort_values("_wc"), "CSW with discourse marker")

    rep = csw[csw["has_repetition"].astype(bool)]
    show(rep.sort_values("_wc"), "CSW with repetition")


if __name__ == "__main__":
    for corpus, (path, l1) in FILES.items():
        run(corpus, path, l1)
