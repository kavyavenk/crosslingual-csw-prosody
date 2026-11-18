#!/usr/bin/env python3
"""
Diagnostic CS vs monolingual classifier (Bhattacharya et al. 2026 style).

Requires Bangor Miami DisVoice features at:
  features/bangor_miami_disvoice_utt.csv  (or pass --train-features)

If absent, trains an in-corpus SEAME speaker-disjoint baseline only and writes
a TODO block for zero-shot SEAME→MaSaC / BM→SEAME transfer.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paper_feature_selection import prosody_feature_columns


def load_xy(df: pd.DataFrame, zscore_within_speaker: bool = False, feats=None):
    y = (df["condition"] == "code_switched").astype(int).values
    sp = df["speaker_id"].values if "speaker_id" in df.columns else df["speaker"].values
    if feats is None:
        feats = prosody_feature_columns(df)
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    if zscore_within_speaker:
        X = X.copy()
        for s in np.unique(sp):
            m = sp == s
            mu = X.loc[m].mean()
            sd = X.loc[m].std().replace(0, 1)
            X.loc[m] = (X.loc[m] - mu) / sd
    # any column that was entirely NaN (median==NaN) or zscored to NaN -> 0
    X = X.fillna(0.0)
    return X.values, y, sp, feats


def speaker_disjoint_split(sp, test_size=0.2, seed=0):
    uniq = np.unique(sp)
    tr, te = train_test_split(uniq, test_size=test_size, random_state=seed)
    return np.isin(sp, tr), np.isin(sp, te)


def speaker_col(df: pd.DataFrame) -> str:
    return "speaker_id" if "speaker_id" in df.columns else "speaker"


def run_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    zscore_train: bool,
    zscore_test: bool,
    seeds: list[int],
) -> dict:
    rows = []
    sp_col = speaker_col(train_df)
    # Shared feature space: align train/test to their common DisVoice columns,
    # ordered by the train corpus, so the fitted model applies to identical columns.
    train_feats = prosody_feature_columns(train_df)
    test_feats = set(prosody_feature_columns(test_df))
    feats = [c for c in train_feats if c in test_feats]
    for seed in seeds:
        tr_m, te_m = speaker_disjoint_split(train_df[sp_col].values, seed=seed)
        X_tr, y_tr, _, _ = load_xy(train_df.iloc[tr_m], zscore_within_speaker=zscore_train, feats=feats)
        X_te, y_te, _, _ = load_xy(test_df, zscore_within_speaker=zscore_test, feats=feats)
        _, te_hold = speaker_disjoint_split(train_df[sp_col].values, seed=seed)
        X_hold, y_hold, _, _ = load_xy(train_df.iloc[te_hold], zscore_within_speaker=zscore_train, feats=feats)
        scaler = StandardScaler().fit(X_tr)
        clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced", random_state=seed)
        clf.fit(scaler.transform(X_tr), y_tr)
        for name, X, y in [("train_corpus_holdout", X_hold, y_hold), ("zero_shot", X_te, y_te)]:
            if len(np.unique(y)) < 2:
                continue
            prob = clf.predict_proba(scaler.transform(X))[:, 1]
            pred = (prob >= 0.5).astype(int)
            rows.append(
                {
                    "seed": seed,
                    "eval": name,
                    "auroc": float(roc_auc_score(y, prob)),
                    "accuracy": float(accuracy_score(y, pred)),
                    "f1": float(f1_score(y, pred, zero_division=0)),
                    "n": len(y),
                    "zscore_train": zscore_train,
                    "zscore_test": zscore_test,
                }
            )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-features", default="features/bangor_miami_disvoice_utt.csv")
    p.add_argument(
        "--seame",
        default="results/functional_anchors/seame/seame_integrated_features.csv",
    )
    p.add_argument(
        "--masac",
        default="results/functional_anchors/masac/masac_integrated_features.csv",
    )
    p.add_argument("--out-dir", default="results/paper/diagnostic_classifier")
    p.add_argument("--seeds", default="0,1,2,3,4")
    args = p.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    train_path = pathlib.Path(args.train_features)
    seame_path = pathlib.Path(args.seame)
    masac_path = pathlib.Path(args.masac)

    all_rows = []
    note = ""

    if train_path.exists():
        train_df = pd.read_csv(train_path, low_memory=False)
        train_label = "bangor_miami"
    elif seame_path.exists():
        train_df = pd.read_csv(seame_path, low_memory=False)
        train_label = "seame_in_corpus_baseline"
        note = (
            "Bangor Miami features not found in repo; report in-corpus SEAME "
            "speaker-disjoint baseline only. Add features/bangor_miami_disvoice_utt.csv "
            "for cross-corpus zero-shot per Bhattacharya et al. (2026)."
        )
    else:
        note = "No training features found."
        (out / "README.md").write_text(note, encoding="utf-8")
        print(note)
        return

    for ztr, zte, label in [
        (False, False, "raw"),
        (True, True, "zscore_within_speaker"),
    ]:
        test_targets = []
        if train_label != "seame_in_corpus_baseline" and seame_path.exists():
            test_targets.append(("SEAME", pd.read_csv(seame_path, low_memory=False)))
        if train_label != "masac" and masac_path.exists():
            test_targets.append(("MaSaC", pd.read_csv(masac_path, low_memory=False)))
        if train_label == "seame_in_corpus_baseline":
            test_targets = []  # holdout only; no zero-shot on train corpus
        for tname, tdf in test_targets:
            rows = run_eval(train_df, tdf, ztr, zte, seeds)
            for r in rows:
                if r["eval"] == "zero_shot" and train_label in str(tname).lower():
                    continue
            for r in rows:
                r["featureization"] = label
                r["train_corpus"] = train_label
                r["test_corpus"] = tname
            all_rows.extend(rows)

    res = pd.DataFrame(all_rows)
    res.to_csv(out / "diagnostic_classifier_results.csv", index=False)
    if not res.empty:
        summary = (
            res.groupby(["train_corpus", "test_corpus", "eval", "featureization"])[
                ["auroc", "accuracy", "f1"]
            ]
            .agg(["mean", "std"])
            .reset_index()
        )
        summary.columns = [
            "_".join(c).rstrip("_") for c in summary.columns.to_flat_index()
        ]
        summary.to_csv(out / "diagnostic_classifier_summary.csv", index=False)
    meta = {"note": note, "seeds": seeds}
    (out / "diagnostic_classifier_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out}; note={note or 'ok'}")


if __name__ == "__main__":
    main()
