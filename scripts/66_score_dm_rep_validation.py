#!/usr/bin/env python3
"""
Score the MaSaC discourse-marker (DM) / repetition (Rep) substring detectors
against human labels produced with scripts/65_build_dm_rep_validation.py.

Treats the HUMAN label as ground truth and the AUTO (substring) label as the
"prediction", so:
  precision = of utterances the detector flags, share truly positive
  recall    = of truly positive utterances, share the detector catches

Because the sample is balanced across DM x Rep cells (not corpus-prevalence),
we also report prevalence-reweighted precision/recall/accuracy using the corpus
base rates saved by the build script, giving an unbiased estimate for MaSaC.

Usage
-----
python3 scripts/66_score_dm_rep_validation.py
python3 scripts/66_score_dm_rep_validation.py \
    --filled results/paper/validation/masac_dm_rep_validation_BLANK.csv
"""
import argparse
import pathlib
import numpy as np
import pandas as pd

VAL_DIR = pathlib.Path("results/paper/validation")

TRUE_TOKENS = {"1", "1.0", "yes", "y", "true", "t"}
FALSE_TOKENS = {"0", "0.0", "no", "n", "false", "f"}


def to_bool(series: pd.Series) -> pd.Series:
    def conv(v):
        s = str(v).strip().lower()
        if s in TRUE_TOKENS:
            return True
        if s in FALSE_TOKENS:
            return False
        return np.nan
    return series.map(conv)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else float("nan")
    acc = (tp + tn) / n if n else float("nan")
    # Cohen's kappa
    po = acc
    p_yes = ((tp + fp) / n) * ((tp + fn) / n)
    p_no = ((tn + fn) / n) * ((tn + fp) / n)
    pe = p_yes + p_no
    kappa = (po - pe) / (1 - pe) if (1 - pe) else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, n=n,
                precision=prec, recall=rec, f1=f1, accuracy=acc, kappa=kappa)


def prevalence_weighted(y_true, y_pred, base_rate):
    """Reweight positive/negative rows so the positive share == corpus base_rate."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pos = y_true
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    w = np.where(pos, base_rate / n_pos, (1 - base_rate) / n_neg)
    tp = float(np.sum(w[y_true & y_pred]))
    fp = float(np.sum(w[~y_true & y_pred]))
    fn = float(np.sum(w[y_true & ~y_pred]))
    tn = float(np.sum(w[~y_true & ~y_pred]))
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else float("nan")
    acc = tp + tn
    return dict(precision=prec, recall=rec, f1=f1, accuracy=acc)


def report(name, m, mw):
    print(f"\n=== {name} ===")
    print(f"  n labeled: {m['n']}   confusion [tp fp fn tn] = "
          f"[{m['tp']} {m['fp']} {m['fn']} {m['tn']}]")
    print(f"  precision {m['precision']:.3f}  recall {m['recall']:.3f}  "
          f"F1 {m['f1']:.3f}  acc {m['accuracy']:.3f}  kappa {m['kappa']:.3f}")
    if mw:
        print(f"  [prevalence-weighted] precision {mw['precision']:.3f}  "
              f"recall {mw['recall']:.3f}  F1 {mw['f1']:.3f}  acc {mw['accuracy']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filled", default=str(VAL_DIR / "masac_dm_rep_validation_BLANK.csv"),
                    help="the sheet with human_dm/human_rep filled in")
    ap.add_argument("--key", default=str(VAL_DIR / "masac_dm_rep_validation_KEY.csv"))
    ap.add_argument("--baserates", default=str(VAL_DIR / "masac_corpus_baserates.csv"))
    args = ap.parse_args()

    filled = pd.read_csv(args.filled, dtype=str)
    key = pd.read_csv(args.key)
    br = pd.read_csv(args.baserates).iloc[0]

    filled["human_dm"] = to_bool(filled["human_dm"])
    filled["human_rep"] = to_bool(filled["human_rep"])

    n_missing = filled["human_dm"].isna().sum() + filled["human_rep"].isna().sum()
    if n_missing:
        print(f"WARNING: {int(n_missing)} blank/unrecognized human cells; "
              "those rows are dropped per-metric. Fill with 1/0 (or yes/no).")

    merged = filled.merge(key, on="utt_id", how="inner", suffixes=("", "_key"))
    if len(merged) != len(filled):
        print(f"NOTE: {len(filled)-len(merged)} rows did not match the key on utt_id.")

    dm = merged.dropna(subset=["human_dm"])
    m_dm = metrics(dm["human_dm"].to_numpy(bool),
                   dm["has_discourse_marker"].to_numpy(bool))
    mw_dm = prevalence_weighted(dm["human_dm"].to_numpy(bool),
                                dm["has_discourse_marker"].to_numpy(bool), br["dm_rate"])

    rep = merged.dropna(subset=["human_rep"])
    m_rep = metrics(rep["human_rep"].to_numpy(bool),
                    rep["has_repetition"].to_numpy(bool))
    mw_rep = prevalence_weighted(rep["human_rep"].to_numpy(bool),
                                 rep["has_repetition"].to_numpy(bool), br["rep_rate"])

    print(f"Corpus base rates: DM={br['dm_rate']:.3f}, Rep={br['rep_rate']:.3f} "
          f"(N={int(br['n_corpus'])})")
    report("Discourse markers (auto vs. human)", m_dm, mw_dm)
    report("Repetition (auto vs. human)", m_rep, mw_rep)

    # Disagreements dumped for qualitative inspection.
    dm_bad = dm[dm["human_dm"].to_numpy(bool) != dm["has_discourse_marker"].to_numpy(bool)]
    rep_bad = rep[rep["human_rep"].to_numpy(bool) != rep["has_repetition"].to_numpy(bool)]
    out = VAL_DIR / "disagreements.csv"
    cols = ["utt_id", "lang", "condition", "text",
            "human_dm", "has_discourse_marker", "human_rep", "has_repetition"]
    cols = [c for c in cols if c in merged.columns]
    pd.concat([dm_bad, rep_bad]).drop_duplicates("utt_id")[cols].to_csv(out, index=False,
                                                                        encoding="utf-8-sig")
    print(f"\nDisagreements ({len(dm_bad)} DM, {len(rep_bad)} Rep) -> {out}")

    print("\n--- copy-paste summary ---")
    print(
        f"On a stratified sample of {m_dm['n']} MaSaC utterances hand-labeled by a "
        f"Hindi-speaking author, the substring discourse-marker detector reached "
        f"precision {m_dm['precision']:.2f}, recall {m_dm['recall']:.2f}, "
        f"F1 {m_dm['f1']:.2f} (Cohen's kappa {m_dm['kappa']:.2f}); the repetition "
        f"detector reached precision {m_rep['precision']:.2f}, recall "
        f"{m_rep['recall']:.2f}, F1 {m_rep['f1']:.2f} (kappa {m_rep['kappa']:.2f}). "
        f"Prevalence-weighted to MaSaC base rates, DM precision/recall are "
        f"{mw_dm['precision']:.2f}/{mw_dm['recall']:.2f} and repetition "
        f"{mw_rep['precision']:.2f}/{mw_rep['recall']:.2f}."
    )


if __name__ == "__main__":
    main()
