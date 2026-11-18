"""
Section 5.3 reframed analyses: speaker + interlocutor vs monolingual baselines.

Mirrors the gender framing (prop_sig over 103 features + optional OLS supplement):
  Speaker: gender, age, nationality
  Interlocutor match (prev speaker): gender, age, nationality

Monolingual baseline = pooled monolingual EN + monolingual L1 within each corpus.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

import paper_feature_selection as pfs

ALPHA = 0.05


def canonical_103():
    return pd.read_csv("features/first_contrast_seame_CS_vs_EN.csv")["feature"].tolist()


def cohend(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp if sp else np.nan


def prop_sig(sub, mono, feats):
    pvals, ds = [], []
    for f in feats:
        a = sub[f].dropna().values
        b = mono[f].dropna().values
        if len(a) < 3 or len(b) < 3:
            continue
        _, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        if not np.isnan(p):
            pvals.append(p)
            d = cohend(a, b)
            if not np.isnan(d):
                ds.append(abs(d))
    if not pvals:
        return 0, 0, float("nan")
    rej = multipletests(pvals, alpha=ALPHA, method="fdr_bh")[0]
    return int(rej.sum()), len(pvals), float(np.mean(ds)) if ds else float("nan")


def compare_groups(cs, mono, feats, g1_mask, g2_mask, l1, l2):
    k1, n1, d1 = prop_sig(cs[g1_mask], mono, feats)
    k2, n2, d2 = prop_sig(cs[g2_mask], mono, feats)
    p1, p2 = k1 / n1 if n1 else 0, k2 / n2 if n2 else 0
    z, pz = proportions_ztest([k1, k2], [n1, n2])
    return dict(
        label1=l1, label2=l2, p1=p1, p2=p2, k1=k1, n1=n1, k2=k2, n2=n2,
        d1=d1, d2=d2, delta=p1 - p2, delta_d=d1 - d2, z=z, pz=pz,
        n1_utt=int(g1_mask.sum()), n2_utt=int(g2_mask.sum()),
    )


def add_match(df, attr, interaction_col="interaction_id"):
    df = df.copy()
    if interaction_col not in df.columns:
        interaction_col = "file_id"
    df["speaker"] = df["speaker"].astype(str)
    if attr == "age_bucket":
        df[attr] = norm_age(df[attr])
    else:
        df[attr] = df[attr].astype(str).str.lower()
    if "start_time" in df.columns:
        df["_s"] = pd.to_numeric(df["start_time"], errors="coerce")
        df = df.sort_values([interaction_col, "_s", "utt_id"], kind="mergesort")
    else:
        parts = df["utt_id"].astype(str).str.rsplit("_", n=1, expand=True)
        df["_s"] = pd.to_numeric(parts[1], errors="coerce")
        df = df.sort_values([interaction_col, "_s", "utt_id"], kind="mergesort")
    df["prev_speaker"] = df.groupby(interaction_col)["speaker"].shift(1)
    spk = (
        df.groupby([interaction_col, "speaker"])[attr]
        .agg(lambda s: s.value_counts().index[0])
        .rename("spk_attr")
        .reset_index()
    )
    df = df.merge(spk, on=[interaction_col, "speaker"], how="left")
    df = df.merge(
        spk.rename(columns={"speaker": "prev_speaker", "spk_attr": "int_attr"}),
        on=[interaction_col, "prev_speaker"],
        how="left",
    )
    valid = {
        "male", "female", "younger", "older",
        "singaporean", "malaysian", "19-24", "25+", "<19",
    }

    def m(r):
        a, b = str(r.spk_attr), str(r.int_attr)
        if a in {"", "nan", "unknown"} or b in {"", "nan", "unknown"}:
            return "unknown"
        if attr == "age_bucket":
            a = "younger" if a in {"19-24", "<19", "younger"} else "older" if a in {"25+", "older"} else "unknown"
            b = "younger" if b in {"19-24", "<19", "younger"} else "older" if b in {"25+", "older"} else "unknown"
            if a == "unknown" or b == "unknown":
                return "unknown"
        elif a not in valid or b not in valid:
            return "unknown"
        return "same" if a == b else "different"

    df["match"] = df.apply(m, axis=1)
    return df


def norm_age(s):
    s = s.astype(str).str.lower()
    out = pd.Series("unknown", index=s.index)
    out[s.isin(["19-24", "<19", "younger"])] = "younger"
    out[s.isin(["25+", "older"])] = "older"
    return out


def speaker_analyses(df, corpus, l1):
    rows = []
    feats = [f for f in canonical_103() if f in df.columns]
    mono = df[df["condition"].isin(["monolingual_EN", f"monolingual_{l1}"])]
    cs = df[df["condition"] == "code_switched"].copy()

    # gender
    cs["gender"] = cs["gender"].astype(str).str.lower()
    r = compare_groups(
        cs, mono, feats,
        cs["gender"] == "female", cs["gender"] == "male",
        "female", "male",
    )
    rows.append(dict(corpus=corpus, level="speaker", dimension="gender", **r))

    # age
    cs["age_norm"] = norm_age(cs["age_bucket"])
    r = compare_groups(
        cs, mono, feats,
        cs["age_norm"] == "younger", cs["age_norm"] == "older",
        "younger", "older",
    )
    rows.append(dict(corpus=corpus, level="speaker", dimension="age", **r))

    # nationality (SEAME only)
    if corpus == "seame":
        cs["nationality"] = cs["nationality"].astype(str).str.lower()
        r = compare_groups(
            cs, mono, feats,
            cs["nationality"] == "singaporean", cs["nationality"] == "malaysian",
            "singaporean", "malaysian",
        )
        rows.append(dict(corpus=corpus, level="speaker", dimension="nationality", **r))
    return rows


def interlocutor_analyses(df, corpus, l1):
    rows = []
    feats = [f for f in canonical_103() if f in df.columns]
    mono = df[df["condition"].isin(["monolingual_EN", f"monolingual_{l1}"])]
    df = df.copy()
    dims = ["gender"]
    if corpus == "seame":
        dims += ["nationality", "age_bucket"]
    else:
        dims += ["age_bucket"]

    for dim in dims:
        d = add_match(df, dim)
        cs = d[(d["condition"] == "code_switched") & d["prev_speaker"].notna()]
        diff = cs["match"] == "different"
        same = cs["match"] == "same"
        r = compare_groups(cs, mono, feats, diff, same, "different", "same")
        rows.append(dict(corpus=corpus, level="interlocutor", dimension=dim, **r))
    return rows


def ols_supplement(cs, dimension, ref, other):
    """CS-only OLS on top-10 non-tilt F0; return n_sig/total and example beta."""
    feats = [f for f in canonical_103() if f in cs.columns]
    f0 = [c for c in feats if pfs.disvoice_category(c) == "f0" and "tilt" not in c.lower()]
    top = cs[f0].var().sort_values(ascending=False).index[:10].tolist()
    sub = cs[cs[dimension].isin([ref, other])].copy()
    rows = []
    for f in top:
        m = sub[[f, dimension]].replace([np.inf, -np.inf], np.nan).dropna()
        if m[f].std() == 0:
            continue
        lhs = f if f.isidentifier() else f'Q("{f}")'
        r = smf.ols(f'{lhs} ~ C({dimension}, Treatment(reference="{ref}"))', m).fit()
        term = [t for t in r.params.index if t.startswith(f"C({dimension}")][0]
        rows.append((f, float(r.params[term]), float(r.pvalues[term])))
    if not rows:
        return 0, 0, None
    res = pd.DataFrame(rows, columns=["feature", "beta", "p"])
    res["sig"] = multipletests(res["p"], alpha=ALPHA, method="fdr_bh")[0]
    ex = res.loc[res["p"].idxmin()]
    return int(res["sig"].sum()), len(res), ex


def print_row(r):
    print(
        f"  {r['level']:11s} {r['dimension']:12s} | "
        f"{r['label1']:10s} {100*r['p1']:5.1f}% ({r['k1']}/{r['n1']}) d={r['d1']:.2f} n={r['n1_utt']} | "
        f"{r['label2']:10s} {100*r['p2']:5.1f}% ({r['k2']}/{r['n2']}) d={r['d2']:.2f} n={r['n2_utt']} | "
        f"Δ={100*r['delta']:+.1f}pp (p={r['pz']:.2g}) Δ|d|={r['delta_d']:+.2f}"
    )


if __name__ == "__main__":
    seame = pd.read_csv(
        "results/functional_anchors/seame/seame_integrated_features.csv", low_memory=False
    )
    masac = pd.read_csv(
        "results/interlocutor_gender/masac_dyad/masac_integrated_features_dyad.csv",
        low_memory=False,
    )
    all_rows = []
    for df, corpus, l1 in [(seame, "seame", "ZH"), (masac, "masac", "HI")]:
        print(f"\n================= {corpus.upper()} =================")
        sp = speaker_analyses(df, corpus, l1)
        il = interlocutor_analyses(df, corpus, l1)
        for r in sp + il:
            print_row(r)
        all_rows.extend(sp + il)

        cs = df[df["condition"] == "code_switched"].copy()
        cs["gender"] = cs["gender"].astype(str).str.lower()
        cs["age_norm"] = norm_age(cs["age_bucket"])
        print("  OLS supplements (top-10 non-tilt F0, CS-only):")
        for dim, ref, other in [
            ("gender", "female", "male"),
            ("age_norm", "younger", "older"),
        ]:
            ns, nt, ex = ols_supplement(cs, dim, ref, other)
            if ex is not None:
                print(f"    {dim}: {ns}/{nt} sig; e.g. {ex['feature']} beta={ex['beta']:+.1f}")
        if corpus == "seame":
            cs["nationality"] = cs["nationality"].astype(str).str.lower()
            ns, nt, ex = ols_supplement(cs, "nationality", "singaporean", "malaysian")
            if ex is not None:
                print(f"    nationality: {ns}/{nt} sig; e.g. {ex['feature']} beta={ex['beta']:+.1f}")

    out = pd.DataFrame(all_rows)
    out.to_csv("results/paper/social_vs_mono_5_3.csv", index=False)
    print("\nSaved -> results/paper/social_vs_mono_5_3.csv")
