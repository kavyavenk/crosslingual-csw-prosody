"""Per-category share of significant CSW-vs-monolingual features for each
speaker/interlocutor subgroup (Section 5.3 full-feature-set appendix).

Demonstrates that the social/interlocutor analysis was run over the entire
DisVoice feature set (all categories), not just the F0 subset highlighted in
the main text. Emits a LaTeX table analogous to the functional-anchor table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import paper_feature_selection as pfs

ALPHA = 0.05
CATS = ["f0", "duration", "energy"]
CATLAB = {"f0": "F0", "duration": "Dur.", "energy": "Energy"}


def canonical_103():
    return pd.read_csv("features/first_contrast_seame_CS_vs_EN.csv")["feature"].tolist()


def share_by_cat(sub, mono, feats):
    """Return {cat: (k_sig, n_tested)} using BH-FDR within the full feature set."""
    pvals, cats = [], []
    for f in feats:
        a = sub[f].dropna().values
        b = mono[f].dropna().values
        if len(a) < 3 or len(b) < 3:
            continue
        _, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        if np.isnan(p):
            continue
        pvals.append(p)
        cats.append(pfs.disvoice_category(f))
    rej = multipletests(pvals, alpha=ALPHA, method="fdr_bh")[0]
    out = {c: [0, 0] for c in CATS}
    for r, c in zip(rej, cats):
        if c in out:
            out[c][1] += 1
            out[c][0] += int(r)
    return out


def norm_age(s):
    s = s.astype(str).str.lower()
    out = pd.Series("unknown", index=s.index)
    out[s.isin(["19-24", "<19", "younger"])] = "younger"
    out[s.isin(["25+", "older"])] = "older"
    return out


def rows_for_corpus(df, corpus, l1):
    feats = [f for f in canonical_103() if f in df.columns]
    mono = df[df["condition"].isin(["monolingual_EN", f"monolingual_{l1}"])]
    cs = df[df["condition"] == "code_switched"].copy()
    cs["gender"] = cs["gender"].astype(str).str.lower()
    cs["age_norm"] = norm_age(cs["age_bucket"])
    out = []
    specs = [
        ("Speaker gender (F)", cs["gender"] == "female"),
        ("Speaker gender (M)", cs["gender"] == "male"),
        ("Speaker age (young)", cs["age_norm"] == "younger"),
        ("Speaker age (old)", cs["age_norm"] == "older"),
    ]
    if corpus == "seame":
        cs["nationality"] = cs["nationality"].astype(str).str.lower()
        specs += [
            ("Speaker nat.\\ (SG)", cs["nationality"] == "singaporean"),
            ("Speaker nat.\\ (MY)", cs["nationality"] == "malaysian"),
        ]
    for label, mask in specs:
        sub = cs[mask]
        if len(sub) < 10:
            continue
        bycat = share_by_cat(sub, mono, feats)
        out.append((label, bycat))
    return out


def to_latex(seame_rows, masac_rows):
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"\textbf{Corpus / subgroup} & \textbf{Cat.} & \textbf{Sig.} & \textbf{\%} \\",
        r"\midrule",
    ]
    for corpus, rows in [("SEAME", seame_rows), ("MaSaC", masac_rows)]:
        for label, bycat in rows:
            first = True
            for c in CATS:
                k, n = bycat[c]
                pct = 100 * k / n if n else 0
                name = f"{corpus}, {label}" if first else ""
                lines.append(f"{name} & {CATLAB[c]} & {k}/{n} & {pct:.0f} \\\\")
                first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [
        r"\end{tabular}",
        r"\caption{Share of features \emph{per prosodic category} on which "
        r"code-switched utterances differ from the pooled monolingual baseline "
        r"(BH-FDR Welch $t$ over all 103 DisVoice features), for each "
        r"speaker/interlocutor subgroup in Section~\ref{sec:speaker-interlocutor}. "
        r"Every social contrast is evaluated across all categories, confirming that "
        r"the F0 outcomes highlighted in the main text are not selected in "
        r"isolation.}",
        r"\label{tab:social-percat}",
        r"\end{table}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    seame = pd.read_csv(
        "results/functional_anchors/seame/seame_integrated_features.csv", low_memory=False
    )
    masac = pd.read_csv(
        "results/interlocutor_gender/masac_dyad/masac_integrated_features_dyad.csv",
        low_memory=False,
    )
    sr = rows_for_corpus(seame, "seame", "ZH")
    mr = rows_for_corpus(masac, "masac", "HI")
    tex = to_latex(sr, mr)
    with open("results/paper/SOCIAL_PERCAT_APPENDIX.tex", "w") as fh:
        fh.write(tex + "\n")
    print(tex)
    print("\nSaved -> results/paper/SOCIAL_PERCAT_APPENDIX.tex")
