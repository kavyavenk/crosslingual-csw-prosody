"""
Full Section 5.3 reframe (parallels the speaker-gender reframe in script 56).

For every social dimension we compute, per corpus, the proportion of the 103
DisVoice features on which a code-switched subgroup differs significantly
(Welch t, BH-FDR) from POOLED monolingual speech (EN + Mandarin/Hindi), with
mean |Cohen's d|; subgroup contrasts tested with a two-proportion z-test.

SPEAKER-CENTRIC (subgroup = CS utterances by that speaker type):
  - gender:      female vs male          (SEAME, MaSaC)
  - age:         younger vs older        (SEAME 19-24 vs 25+; MaSaC younger vs older)
  - nationality: Singaporean vs Malaysian (SEAME only; MaSaC all Indian)

INTERLOCUTOR-MATCH (subgroup = CS turns where interlocutor = prev speaker is
SAME vs DIFFERENT on that dimension):
  - gender  (SEAME, MaSaC)
  - age     (SEAME 19-24/25+, MaSaC younger/older)
  - nationality (SEAME only)
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


def cohend(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp if sp else np.nan


def prop_sig(sub, mono, feats):
    ps, ds = [], []
    for f in feats:
        a = sub[f].dropna().values
        b = mono[f].dropna().values
        if len(a) < 3 or len(b) < 3:
            continue
        _, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        if not np.isnan(p):
            ps.append(p)
            d = cohend(a, b)
            if not np.isnan(d):
                ds.append(abs(d))
    if not ps:
        return 0, 0, float("nan")
    rej = multipletests(ps, alpha=ALPHA, method="fdr_bh")[0]
    return int(rej.sum()), len(ps), float(np.mean(ds)) if ds else float("nan")


def contrast(sub_a, sub_b, mono, feats, la, lb, tag, rows, corpus):
    ka, na, da = prop_sig(sub_a, mono, feats)
    kb, nb, db = prop_sig(sub_b, mono, feats)
    pa, pb = ka / na if na else 0, kb / nb if nb else 0
    try:
        z, pz = proportions_ztest([ka, kb], [na, nb])
    except Exception:
        z, pz = np.nan, np.nan
    print(f"  {tag:26s} {la} {100*pa:5.1f}% ({ka}/{na}) d={da:.2f} n={len(sub_a)} | "
          f"{lb} {100*pb:5.1f}% ({kb}/{nb}) d={db:.2f} n={len(sub_b)} | "
          f"Δ={100*(pa-pb):+5.1f}pp (p={pz:.2g}) Δ|d|={da-db:+.2f}")
    rows.append(dict(corpus=corpus, contrast=tag, a=la, b=lb, pa=pa, pb=pb,
                     ka=ka, na=na, kb=kb, nb=nb, da=da, db=db,
                     delta_pp=100*(pa-pb), delta_d=da-db, na_utt=len(sub_a),
                     nb_utt=len(sub_b), z=z, pz=pz))


# ---------- interlocutor matching ----------
def add_match(df, attr, interaction_col="interaction_id"):
    df = df.copy()
    if interaction_col not in df.columns:
        df[interaction_col] = df["file_id"].astype(str)
    df["speaker"] = df["speaker"].astype(str)
    df[attr] = df[attr].astype(str).str.lower()
    if "start_time" in df.columns:
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
    valid = {"male", "female", "younger", "older", "19-24", "25+",
             "singaporean", "malaysian"}

    def m(r):
        a, b = str(r.spk_attr), str(r.int_attr)
        if a not in valid or b not in valid:
            return "unknown"
        return "same" if a == b else "different"
    df["match"] = df.apply(m, axis=1)
    return df


def norm_age(s):
    s = str(s).lower()
    if s in ("19-24", "younger", "<19"):
        return "younger"
    if s in ("25+", "older"):
        return "older"
    return "other"


def run(df, corpus, l1, dims):
    print(f"\n================= {corpus.upper()} =================")
    feats = [f for f in canonical_103() if f in df.columns]
    df = df.copy()
    df["gender"] = df["gender"].astype(str).str.lower()
    df["nationality"] = df["nationality"].astype(str).str.lower()
    df["age_norm"] = df["age_bucket"].map(norm_age)
    mono = df[df["condition"].isin(["monolingual_EN", f"monolingual_{l1}"])]
    cs = df[df["condition"] == "code_switched"]
    rows = []

    print("[SPEAKER-CENTRIC]")
    if "gender" in dims:
        contrast(cs[cs.gender == "female"], cs[cs.gender == "male"], mono, feats,
                 "F", "M", "speaker_gender", rows, corpus)
    if "age" in dims:
        contrast(cs[cs.age_norm == "younger"], cs[cs.age_norm == "older"], mono, feats,
                 "young", "old", "speaker_age", rows, corpus)
    if "nationality" in dims:
        contrast(cs[cs.nationality == "singaporean"], cs[cs.nationality == "malaysian"],
                 mono, feats, "SG", "MY", "speaker_nationality", rows, corpus)

    print("[INTERLOCUTOR SAME vs DIFFERENT]")
    int_map = {"gender": "gender", "age": "age_norm", "nationality": "nationality"}
    for dim in dims:
        col = int_map[dim]
        d = add_match(df, col)
        csd = d[(d["condition"] == "code_switched") & (d["prev_speaker"].notna())]
        diff = csd[csd["match"] == "different"]
        same = csd[csd["match"] == "same"]
        if len(diff) < 30 or len(same) < 30:
            print(f"  interloc_{dim:18s} skipped (diff={len(diff)}, same={len(same)})")
            continue
        contrast(diff, same, mono, feats, "diff", "same", f"interloc_{dim}", rows, corpus)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    seame = pd.read_csv("results/functional_anchors/seame/seame_integrated_features.csv", low_memory=False)
    masac = pd.read_csv("results/interlocutor_gender/masac_dyad/masac_integrated_features_dyad.csv", low_memory=False)
    out = []
    out.append(run(seame, "seame", "ZH", ["gender", "age", "nationality"]))
    out.append(run(masac, "masac", "HI", ["gender", "age"]))
    pd.concat(out, ignore_index=True).to_csv("results/paper/full_5_3_reframe.csv", index=False)
    print("\nSaved -> results/paper/full_5_3_reframe.csv")
