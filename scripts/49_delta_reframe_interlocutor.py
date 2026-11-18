"""
DB-thesis-style reframing of Section 5.3 (interlocutor / audience effects).

For each social dimension (gender, age, nationality), define the interlocutor
as the previous speaker in the interaction and label each turn as SAME vs
DIFFERENT on that dimension. Then, mirroring the thesis framing
("mixed-gender pairs differ MORE from monolingual speech than same-gender ones:
delta = X, z-test p<0.05"), we compute for each subgroup the proportion of the
103 DisVoice features that differ significantly (BH-FDR) from pooled monolingual
speech, and compare DIFFERENT vs SAME via a two-proportion z-test.

SEAME: gender + nationality (age excluded: concentrated age distribution).
MaSaC: gender + age (nationality constant).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

ALPHA = 0.05


def canonical_103():
    fc = pd.read_csv("features/first_contrast_seame_CS_vs_EN.csv")
    return fc["feature"].tolist()


def add_match(df, attr, interaction_col="interaction_id", order_col=None):
    df = df.copy()
    if interaction_col not in df.columns:
        df[interaction_col] = df["file_id"].astype(str)
    df["speaker"] = df["speaker"].astype(str)
    df[attr] = df[attr].astype(str).str.lower()
    if order_col is not None:
        df["_o"] = pd.to_numeric(df[order_col], errors="coerce")
        df = df.sort_values([interaction_col, "_o", "utt_id"], kind="mergesort")
    elif "start_time" in df.columns:
        df["_s"] = pd.to_numeric(df["start_time"], errors="coerce")
        df = df.sort_values([interaction_col, "_s", "utt_id"], kind="mergesort")
    else:
        parts = df["utt_id"].astype(str).str.rsplit("_", n=1, expand=True)
        df["_o"] = pd.to_numeric(parts[1], errors="coerce")
        df = df.sort_values([interaction_col, "_o", "utt_id"], kind="mergesort")
    df["prev_speaker"] = df.groupby(interaction_col)["speaker"].shift(1)
    spk = (df.groupby([interaction_col, "speaker"])[attr]
             .agg(lambda s: s.value_counts().index[0]).rename("spk_attr").reset_index())
    df = df.merge(spk, on=[interaction_col, "speaker"], how="left")
    df = df.merge(spk.rename(columns={"speaker": "prev_speaker", "spk_attr": "int_attr"}),
                  on=[interaction_col, "prev_speaker"], how="left")
    valid = {"male", "female", "younger", "older", "singaporean", "malaysian"}

    def m(r):
        a, b = str(r.spk_attr), str(r.int_attr)
        if a not in valid or b not in valid:
            return "unknown"
        return "same" if a == b else "different"
    df["match"] = df.apply(m, axis=1)
    return df


def _cohend(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp if sp else np.nan


def prop_sig(cs_sub, mono, feats):
    pvals, ds = [], []
    for f in feats:
        a = cs_sub[f].dropna().values
        b = mono[f].dropna().values
        if len(a) < 3 or len(b) < 3:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        if not np.isnan(p):
            pvals.append(p)
            d = _cohend(a, b)
            if not np.isnan(d):
                ds.append(abs(d))
    if not pvals:
        return 0, 0, float("nan")
    rej, _, _, _ = multipletests(pvals, alpha=ALPHA, method="fdr_bh")
    return int(rej.sum()), len(pvals), float(np.mean(ds)) if ds else float("nan")


def run(df, corpus, attrs, l1):
    print(f"\n================= {corpus.upper()} =================")
    feats = [f for f in canonical_103() if f in df.columns]
    mono = df[df["condition"].isin(["monolingual_EN", f"monolingual_{l1}"])]
    rows = []
    for attr, order_col in attrs:
        d = add_match(df, attr, order_col=order_col)
        cs = d[(d["condition"] == "code_switched") & (d["prev_speaker"].notna())]
        diff = cs[cs["match"] == "different"]
        same = cs[cs["match"] == "same"]
        kd, nd, dd = prop_sig(diff, mono, feats)
        ks, ns, dsame = prop_sig(same, mono, feats)
        pd_ = kd / nd if nd else 0
        ps_ = ks / ns if ns else 0
        try:
            z, pz = proportions_ztest([kd, ks], [nd, ns])
        except Exception:
            z, pz = np.nan, np.nan
        print(f"  {attr:12s} DIFFERENT {100*pd_:5.1f}% ({kd}/{nd}) d={dd:.2f} n={len(diff)} | "
              f"SAME {100*ps_:5.1f}% ({ks}/{ns}) d={dsame:.2f} n={len(same)} | "
              f"Δ%={100*(pd_-ps_):+5.1f}pp (z p={pz:.2g}) Δ|d|={dd-dsame:+.2f}")
        rows.append(dict(corpus=corpus, attr=attr, p_diff=pd_, p_same=ps_,
                         delta=pd_ - ps_, k_diff=kd, n_diff=nd, k_same=ks, n_same=ns,
                         d_diff=dd, d_same=dsame, delta_d=dd - dsame,
                         n_utt_diff=len(diff), n_utt_same=len(same), z=z, pz=pz))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    seame = pd.read_csv("results/functional_anchors/seame/seame_integrated_features.csv", low_memory=False)
    masac = pd.read_csv("results/interlocutor_gender/masac_dyad/masac_integrated_features_dyad.csv", low_memory=False)
    out = []
    out.append(run(seame, "seame", [("gender", None), ("nationality", None)], "ZH"))
    out.append(run(masac, "masac", [("gender", None), ("age_bucket", None)], "HI"))
    pd.concat(out, ignore_index=True).to_csv("results/paper/delta_reframe_5_3.csv", index=False)
    print("\nSaved -> results/paper/delta_reframe_5_3.csv")
