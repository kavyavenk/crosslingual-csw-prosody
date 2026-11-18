"""
[[Kavya-3]] (3YkG W5) MaSaC exactly-two-speaker filter: how many speakers and
utterances survive, and confirmation that all 5 primary speakers are retained.

[[Kavya-4]] (hoSr typo 7) "All speakers produce all three utterance types":
per-speaker counts of monolingual-EN, monolingual-L1, and CSW, with how many
speakers produce all three (and under what utterance-count threshold).
"""
from __future__ import annotations

import pandas as pd

PRIMARY = ["MAYA", "INDRAVARDHAN", "SAHIL", "MONISHA", "ROSESH"]

FULL = {
    "seame": ("results/functional_anchors/seame/seame_integrated_features.csv", "ZH"),
    "masac": ("results/functional_anchors/masac/masac_integrated_features.csv", "HI"),
}
DYAD = "results/interlocutor_gender/masac_dyad/masac_integrated_features_dyad.csv"


def three_type_table(df, l1, corpus):
    df = df.copy()
    df["cond3"] = df["condition"].map(
        {"code_switched": "CSW", "monolingual_EN": "mono_EN",
         f"monolingual_{l1}": f"mono_{l1}"}
    )
    piv = (
        df.pivot_table(index="speaker_id", columns="cond3",
                       values="utt_id", aggfunc="count", fill_value=0)
    )
    for c in ["CSW", "mono_EN", f"mono_{l1}"]:
        if c not in piv:
            piv[c] = 0
    piv["total"] = piv[["CSW", "mono_EN", f"mono_{l1}"]].sum(axis=1)
    piv["has_all3"] = (piv[["CSW", "mono_EN", f"mono_{l1}"]] > 0).all(axis=1)
    return piv.sort_values("total", ascending=False)


def kavya4(corpus, path, l1):
    df = pd.read_csv(path, low_memory=False)
    piv = three_type_table(df, l1, corpus)
    n_spk = len(piv)
    print(f"\n===== [[Kavya-4]] {corpus.upper()}: three-utterance-type coverage =====")
    print(f"total speakers: {n_spk}")
    print(f"produce ALL three types (>=1 each): {int(piv['has_all3'].sum())}/{n_spk}")
    for thr in [1, 5, 10, 20, 30, 50]:
        sub = piv[piv["total"] >= thr]
        print(f"  speakers with >= {thr:>3d} total utts: {len(sub):>3d}  "
              f"| of those, all-3: {int(sub['has_all3'].sum()):>3d} "
              f"({100*sub['has_all3'].mean():.0f}%)")
    # who lacks a type among the sizeable speakers
    missing = piv[(piv["total"] >= 20) & (~piv["has_all3"])]
    if len(missing):
        print(f"  speakers >=20 utts missing a type:")
        print(missing[["CSW", "mono_EN", f"mono_{l1}", "total"]].to_string())
    piv.to_csv(f"results/paper/validation/kavya4_{corpus}_speaker_types.csv")
    return piv


def kavya3():
    print("\n===== [[Kavya-3]] MaSaC exactly-two-speaker filter =====")
    full = pd.read_csv(FULL["masac"][0], low_memory=False)
    dy = pd.read_csv(DYAD, low_memory=False)

    n_int_full = full["interaction_id"].nunique()
    # replicate the raw exactly-2-speaker count on the FULL file (no exclusions)
    n_spk_by_int = full.groupby("interaction_id")["speaker_id"].nunique()
    raw2 = n_spk_by_int[n_spk_by_int == 2].index
    print(f"full corpus: {len(full)} utts, {full['speaker_id'].nunique()} speakers, "
          f"{n_int_full} interactions")
    print(f"interactions with exactly 2 distinct speakers (raw): {len(raw2)} "
          f"({full['interaction_id'].isin(raw2).sum()} utts)")

    print(f"\nfinal dyad subset (after excluding role/multi-party speaker labels, "
          f"exactly-2-speaker interactions):")
    print(f"  utterances retained: {len(dy)}")
    print(f"  unique speakers retained: {dy['speaker_id'].nunique()}")
    print(f"  interactions retained: {dy['interaction_id'].nunique()}")
    csw = dy[dy["condition"] == "code_switched"]
    print(f"  CSW utterances retained: {len(csw)}")

    present = {p: int((dy["speaker_id"].astype(str).str.upper() == p).sum())
               for p in PRIMARY}
    print(f"\n  primary-speaker utterance counts in dyad subset:")
    for p, c in present.items():
        print(f"    {p:14s}: {c}  {'RETAINED' if c > 0 else 'MISSING'}")
    all_kept = all(v > 0 for v in present.values())
    print(f"  => all 5 primary speakers retained: {all_kept}")


if __name__ == "__main__":
    kavya3()
    for corpus, (path, l1) in FULL.items():
        kavya4(corpus, path, l1)
    print("\nDONE")
