"""
[[Kavya-1]] Reviewer hoSr W3: replace the two-proportion z-tests (which treat the
103 correlated features as independent) with utterance-level HIERARCHICAL models
with by-speaker random intercepts, testing whether CSW status INTERACTS with the
key predictors.

Two model families per corpus, one model per DisVoice feature (BH-FDR across the
103 features within each interaction term):

  Model A  (pooled: CSW + monolingual, tests amplification of the CSW-vs-mono gap)
      feat ~ is_csw * has_discourse_marker + is_csw * has_repetition
             + (1 | speaker)
    -> interaction  is_csw:has_discourse_marker   (DM amplification)
    -> interaction  is_csw:has_repetition          (Rep modulation)
    A significant interaction whose sign MATCHES the is_csw main effect (the raw
    CSW-vs-mono gap) means the predictor WIDENS (amplifies) the gap; an opposite
    sign means it NARROWS (attenuates) it.

  Model B  (CSW-only: richness + strategy are only defined for code-switches, so an
            is_csw x predictor interaction is unidentifiable -- monolingual turns
            have no switches/strategy. We instead fit a by-speaker hierarchical
            model over CSW utterances and read the slope, then compare its sign to
            the CSW-vs-mono gap.)
      feat ~ n_switches + C(strategy2) + has_discourse_marker + has_repetition
             + (1 | speaker)                        [strategy2 in {ins, alt}]
    -> slope  n_switches           (richness)
    -> effect C(strategy2)[ins]    (insertional vs alternational)

Outputs one tidy CSV per corpus/model plus a printed summary of BH-FDR-significant
counts and their direction.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.simplefilter("ignore")
ALPHA = 0.05

PATHS = {
    "seame": ("results/functional_anchors/seame/seame_integrated_features.csv", "ZH"),
    "masac": ("results/functional_anchors/masac/masac_integrated_features.csv", "HI"),
}


def canonical_103():
    return pd.read_csv("features/first_contrast_seame_CS_vs_EN.csv")["feature"].tolist()


def lhs(col):
    return col if col.isidentifier() else 'Q("%s")' % col.replace('"', '\\"')


def _fit(formula, data, group):
    try:
        return smf.mixedlm(formula, data, groups=data[group]).fit(
            method=["lbfgs", "powell"], maxiter=200
        )
    except Exception:
        return None


def _term(names, must_have):
    for n in names:
        if all(k in n for k in must_have):
            return n
    return None


# ----------------------------- Model A (pooled) ----------------------------- #
def model_a(df, feats, group="speaker_id"):
    df = df.copy()
    df["is_csw"] = df["is_csw"].astype(float)
    df["dm"] = df["has_discourse_marker"].astype(int)
    df["rep"] = df["has_repetition"].astype(int)
    rows = []
    for i, feat in enumerate(feats, 1):
        cols = [feat, "is_csw", "dm", "rep", group]
        m = df[cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        if len(m) < 200 or m[feat].std() == 0:
            print(f"  [A {i}/{len(feats)}] {feat}: skipped", flush=True)
            continue
        f = f"{lhs(feat)} ~ is_csw * dm + is_csw * rep"
        r = _fit(f, m, group)
        if r is None:
            print(f"  [A {i}/{len(feats)}] {feat}: fit failed", flush=True)
            continue
        names = list(r.params.index)
        b_csw = float(r.params.get("is_csw", np.nan))
        dm_int = _term(names, ["is_csw:", "dm"])
        rep_int = _term(names, ["is_csw:", "rep"])
        row = dict(feature=feat, n=len(m), b_is_csw=b_csw)
        if dm_int:
            row.update(b_dm_int=float(r.params[dm_int]), p_dm_int=float(r.pvalues[dm_int]))
        if rep_int:
            row.update(b_rep_int=float(r.params[rep_int]), p_rep_int=float(r.pvalues[rep_int]))
        rows.append(row)
        print(f"  [A {i}/{len(feats)}] {feat}: ok", flush=True)
    return pd.DataFrame(rows)


# --------------------------- Model B (CSW-only) ----------------------------- #
def model_b(df, feats, group="speaker_id"):
    cs = df[df["condition"] == "code_switched"].copy()
    cs["dm"] = cs["has_discourse_marker"].astype(int)
    cs["rep"] = cs["has_repetition"].astype(int)
    cs["n_switches"] = pd.to_numeric(cs["n_switches"], errors="coerce")
    s = cs["cs_strategy"].astype(str).str.lower()
    cs["strategy2"] = pd.Series(pd.NA, index=cs.index, dtype="object")
    cs.loc[s.str.startswith("ins"), "strategy2"] = "insertional"
    cs.loc[s.str.startswith("alt"), "strategy2"] = "alternational"
    # pooled monolingual means for reference gap sign
    mono = df[df["condition"] != "code_switched"]
    rows = []
    for i, feat in enumerate(feats, 1):
        sub = cs[[feat, "n_switches", "strategy2", "dm", "rep", group]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna().reset_index(drop=True)
        if len(sub) < 200 or sub[feat].std() == 0:
            print(f"  [B {i}/{len(feats)}] {feat}: skipped", flush=True)
            continue
        gap = float(cs[feat].mean() - mono[feat].mean())  # CSW - mono
        f = (f'{lhs(feat)} ~ n_switches '
             f'+ C(strategy2, Treatment(reference="alternational")) + dm + rep')
        r = _fit(f, sub, group)
        if r is None:
            print(f"  [B {i}/{len(feats)}] {feat}: fit failed", flush=True)
            continue
        names = list(r.params.index)
        strat = _term(names, ["strategy2", "insertional"])
        row = dict(
            feature=feat, n=len(sub), gap_csw_minus_mono=gap,
            b_nswitches=float(r.params.get("n_switches", np.nan)),
            p_nswitches=float(r.pvalues.get("n_switches", np.nan)),
        )
        if strat:
            row.update(b_strategy_ins=float(r.params[strat]),
                       p_strategy_ins=float(r.pvalues[strat]))
        rows.append(row)
        print(f"  [B {i}/{len(feats)}] {feat}: ok", flush=True)
    return pd.DataFrame(rows)


def _bh(df, pcol):
    ok = df[pcol].notna()
    out = pd.Series(False, index=df.index)
    if ok.sum():
        out.loc[ok] = multipletests(df.loc[ok, pcol], alpha=ALPHA, method="fdr_bh")[0]
    return out


def summarize(corpus, a, b):
    print(f"\n########## {corpus.upper()} : Kavya-1 interaction summary ##########")
    # ---- DM amplification: gap grows in MAGNITUDE when DM present ----
    a = a.copy()
    a["sig_dm"] = _bh(a, "p_dm_int")
    a["amplify_dm"] = (a["b_is_csw"] + a["b_dm_int"]).abs() > a["b_is_csw"].abs()
    sd = a[a["sig_dm"]]
    print(f"CSW x DM: {sd.shape[0]}/{a['p_dm_int'].notna().sum()} features BH-FDR sig; "
          f"of these {int(sd['amplify_dm'].sum())} AMPLIFY the CSW-vs-mono gap, "
          f"{int((~sd['amplify_dm']).sum())} attenuate")
    # ---- Rep ----
    a["sig_rep"] = _bh(a, "p_rep_int")
    a["amplify_rep"] = (a["b_is_csw"] + a["b_rep_int"]).abs() > a["b_is_csw"].abs()
    sr = a[a["sig_rep"]]
    print(f"CSW x Rep: {sr.shape[0]}/{a['p_rep_int'].notna().sum()} features BH-FDR sig; "
          f"of these {int(sr['amplify_rep'].sum())} amplify, "
          f"{int((~sr['amplify_rep']).sum())} attenuate")
    # ---- richness ----
    b = b.copy()
    b["sig_nsw"] = _bh(b, "p_nswitches")
    b["diverge_nsw"] = np.sign(b["b_nswitches"]) == np.sign(b["gap_csw_minus_mono"])
    sn = b[b["sig_nsw"]]
    print(f"Richness (n_switches, CSW-only): {sn.shape[0]}/{b['p_nswitches'].notna().sum()} "
          f"features BH-FDR sig; of these {int(sn['diverge_nsw'].sum())} push FURTHER "
          f"from monolingual (more divergent), {int((~sn['diverge_nsw']).sum())} toward")
    # ---- strategy ----
    if "p_strategy_ins" in b:
        b["sig_str"] = _bh(b, "p_strategy_ins")
        ss = b[b["sig_str"]]
        print(f"Strategy (insertional vs alternational, CSW-only): "
              f"{ss.shape[0]}/{b['p_strategy_ins'].notna().sum()} features BH-FDR sig")
    return a, b


def run(corpus):
    path, _ = PATHS[corpus]
    df = pd.read_csv(path, low_memory=False)
    feats = [f for f in canonical_103() if f in df.columns]
    print(f"\n===== {corpus.upper()} (n={len(df)}, {len(feats)} features) =====")

    a_path = f"results/paper/validation/kavya1_modelA_{corpus}.csv"
    b_path = f"results/paper/validation/kavya1_modelB_{corpus}.csv"
    os.makedirs("results/paper/validation", exist_ok=True)

    if os.path.exists(a_path):
        a = pd.read_csv(a_path)
    else:
        a = model_a(df, feats)
        a.to_csv(a_path, index=False)
    if os.path.exists(b_path):
        b = pd.read_csv(b_path)
    else:
        b = model_b(df, feats)
        b.to_csv(b_path, index=False)

    a, b = summarize(corpus, a, b)
    a.to_csv(a_path, index=False)
    b.to_csv(b_path, index=False)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["masac", "seame"]
    for c in which:
        run(c)
    print("\nDONE")
