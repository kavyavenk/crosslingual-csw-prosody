"""Diagnostic classifier ablation by prosodic category (Appendix / §5.1 DB note)."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_feature_selection as pfs

SEEDS = [0, 1, 2, 3, 4]


def speaker_disjoint_split(sp, test_size=0.2, seed=0):
    uniq = np.unique(sp)
    tr, te = train_test_split(uniq, test_size=test_size, random_state=seed)
    return np.isin(sp, tr), np.isin(sp, te)


def load_xy_local(df, zscore_within_speaker=False, feats=None):
    y = (df["condition"] == "code_switched").astype(int).values
    sp = df["speaker_id"].values if "speaker_id" in df.columns else df["speaker"].values
    if feats is None:
        feats = [f for f in pfs.prosody_feature_columns(df) if f in canonical_103()]
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median()).fillna(0.0)
    if zscore_within_speaker:
        X = X.copy()
        for s in np.unique(sp):
            m = sp == s
            mu = X.loc[m].mean()
            sd = X.loc[m].std().replace(0, 1)
            X.loc[m] = (X.loc[m] - mu) / sd
    X = X.fillna(0.0)
    return X.values, y, sp, feats


def canonical_103():
    return pd.read_csv("features/first_contrast_seame_CS_vs_EN.csv")["feature"].tolist()


def feats_by_category(all_feats, drop_cats=()):
    groups = {"f0": [], "duration": [], "energy": [], "other": []}
    for f in all_feats:
        groups[pfs.disvoice_category(f)].append(f)
    keep = []
    for cat, fl in groups.items():
        if cat not in drop_cats:
            keep.extend(fl)
    return keep


def eval_setting(train_df, test_df, feats, z_tr, z_te):
    rows = []
    sp_col = "speaker_id" if "speaker_id" in train_df.columns else "speaker"
    for seed in SEEDS:
        tr_m, _ = speaker_disjoint_split(train_df[sp_col].values, seed=seed)
        _, te_hold = speaker_disjoint_split(train_df[sp_col].values, seed=seed)
        X_tr, y_tr, _, _ = load_xy_local(train_df.iloc[tr_m], zscore_within_speaker=z_tr, feats=feats)
        X_hold, y_hold, _, _ = load_xy_local(train_df.iloc[te_hold], zscore_within_speaker=z_tr, feats=feats)
        X_te, y_te, _, _ = load_xy_local(test_df, zscore_within_speaker=z_te, feats=feats)
        scaler = StandardScaler().fit(X_tr)
        clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced", random_state=seed)
        clf.fit(scaler.transform(X_tr), y_tr)
        for name, X, y in [("BM_holdout", X_hold, y_hold), ("ZS", X_te, y_te)]:
            if len(np.unique(y)) < 2 or len(feats) == 0:
                continue
            prob = clf.predict_proba(scaler.transform(X))[:, 1]
            pred = (prob >= 0.5).astype(int)
            rows.append(
                dict(
                    seed=seed,
                    eval=name,
                    auroc=float(roc_auc_score(y, prob)),
                    accuracy=float(accuracy_score(y, pred)),
                    f1=float(f1_score(y, pred, zero_division=0)),
                    n_feat=len(feats),
                )
            )
    return rows


def summarize(rows, setting, feature_set):
    d = pd.DataFrame(rows)
    g = d.groupby("eval")[["auroc", "accuracy", "f1"]].agg(["mean", "std"])
    out = {}
    for m in ["auroc", "accuracy", "f1"]:
        out[m] = float(g.loc[setting, (m, "mean")])
        out[f"{m}_std"] = float(g.loc[setting, (m, "std")])
    out["feature_set"] = feature_set
    out["eval"] = setting
    out["n_feat"] = rows[0]["n_feat"] if rows else 0
    return out


if __name__ == "__main__":
    bm = pd.read_csv("features/bangor_miami_disvoice_utt.csv", low_memory=False)
    se = pd.read_csv("results/functional_anchors/seame/seame_integrated_features.csv", low_memory=False)
    ma = pd.read_csv("results/functional_anchors/masac/masac_integrated_features.csv", low_memory=False)
    base = [f for f in canonical_103() if f in bm.columns and f in se.columns and f in ma.columns]

    configs = [
        ("all103", base, ()),
        ("no_energy", base, ("energy",)),
        ("no_duration", base, ("duration",)),
        ("no_f0", base, ("f0",)),
        ("f0_only", base, ("duration", "energy", "other")),
        ("duration_only", base, ("f0", "energy", "other")),
        ("energy_only", base, ("f0", "duration", "other")),
    ]
    results = []
    for label, allf, drop in configs:
        feats = feats_by_category(allf, drop)
        print(f"\n=== {label} ({len(feats)} feats) ===")
        for z_tr, z_te, tag in [(False, False, "raw"), (True, True, "z")]:
            rows = eval_setting(bm, se, feats, z_tr, z_te)
            s = summarize(rows, "BM_holdout", label)
            s["norm"] = tag
            s["test"] = "SEAME"
            results.append(s)
            print(f"  BM holdout {tag}: AUROC={s['auroc']:.3f}±{s['auroc_std']:.3f}")
            rows2 = eval_setting(bm, ma, feats, z_tr, z_te)
            s2 = summarize(rows2, "ZS", label)
            s2["norm"] = tag
            s2["test"] = "MaSaC"
            results.append(s2)
            print(f"  ZS MaSaC {tag}: AUROC={s2['auroc']:.3f}±{s2['auroc_std']:.3f}")

    pd.DataFrame(results).to_csv("results/paper/diagnostic_classifier/ablation_by_category.csv", index=False)
    print("\nSaved -> results/paper/diagnostic_classifier/ablation_by_category.csv")
