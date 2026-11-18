#!/usr/bin/env python3
"""
Build a BLIND manual-validation sheet for the MaSaC discourse-marker (DM) and
repetition (Rep) substring detectors.

Rationale
---------
The paper's DM/Rep labels are produced automatically by substring matching
against curated, non-exhaustive inventories (scripts/16_...). Discourse markers
are the strongest discursive predictor, so we validate detector quality on a
human-labeled sample. This script draws a stratified sample (balanced across the
four DM x Rep cells so that both positive and negative classes are well
represented for stable precision/recall) and writes:

  1. a BLANK sheet (no auto labels shown) for the annotator to fill in, and
  2. a hidden KEY holding the auto labels + stratification cell, for scoring.

Fill in the BLANK sheet's `human_dm` and `human_rep` columns (1/0, yes/no,
true/false, y/n all accepted), then run scripts/66_score_dm_rep_validation.py.

Usage
-----
python3 scripts/65_build_dm_rep_validation.py            # defaults: n=325, seed=42
python3 scripts/65_build_dm_rep_validation.py --n 200 --seed 7
"""
import argparse
import pathlib
import pandas as pd

SRC = "results/functional_anchors/masac/masac_integrated_features.csv"
OUT_DIR = pathlib.Path("results/paper/validation")


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Balanced allocation across the 4 (has_discourse_marker, has_repetition) cells."""
    cells = [(False, False), (False, True), (True, False), (True, True)]
    base = n // 4
    rem = n - base * 4
    alloc = {c: base + (1 if i < rem else 0) for i, c in enumerate(cells)}

    parts = []
    for (dm, rep) in cells:
        pool = df[(df["has_discourse_marker"] == dm) & (df["has_repetition"] == rep)]
        take = min(alloc[(dm, rep)], len(pool))
        if take < alloc[(dm, rep)]:
            print(f"  WARN: cell dm={dm} rep={rep} has only {len(pool)} rows; taking {take}")
        parts.append(pool.sample(n=take, random_state=seed))
    sample = pd.concat(parts, ignore_index=True)
    # Shuffle so cells are interleaved (annotator gets no ordering cue).
    return sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=325)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(SRC, low_memory=False)
    df = df[df["text"].notna() & (df["text"].astype(str).str.strip() != "")].copy()

    sample = stratified_sample(df, args.n, args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    blank = sample[["utt_id", "lang", "condition", "text"]].copy()
    blank.insert(0, "row", range(1, len(blank) + 1))
    blank["human_dm"] = ""      # annotator: 1 if a discourse marker is present, else 0
    blank["human_rep"] = ""     # annotator: 1 if a content-word/phrase repetition is present, else 0
    blank["notes"] = ""
    blank_path = OUT_DIR / "masac_dm_rep_validation_BLANK.csv"
    blank.to_csv(blank_path, index=False, encoding="utf-8-sig")

    key = sample[["utt_id", "lang", "condition",
                  "has_discourse_marker", "has_repetition"]].copy()
    key["cell"] = list(zip(sample["has_discourse_marker"], sample["has_repetition"]))
    key_path = OUT_DIR / "masac_dm_rep_validation_KEY.csv"
    key.to_csv(key_path, index=False, encoding="utf-8-sig")

    # Corpus base rates for later prevalence-reweighting in scoring.
    rates = {
        "dm_rate": float(df["has_discourse_marker"].mean()),
        "rep_rate": float(df["has_repetition"].mean()),
        "n_corpus": int(len(df)),
    }
    pd.DataFrame([rates]).to_csv(OUT_DIR / "masac_corpus_baserates.csv", index=False)

    print(f"Wrote {len(blank)} utterances.")
    print(f"  BLANK sheet -> {blank_path}")
    print(f"  KEY (hidden) -> {key_path}")
    print(f"  base rates   -> {OUT_DIR/'masac_corpus_baserates.csv'}")
    print("\nCell counts in sample (auto labels):")
    print(pd.crosstab(sample["has_discourse_marker"], sample["has_repetition"]))
    print("\nLanguage mix in sample:")
    print(sample["lang"].value_counts())


if __name__ == "__main__":
    main()
