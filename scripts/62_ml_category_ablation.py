"""
DB request: ablate each prosodic category from the diagnostic classifier
(Table 4) one at a time and report zero-shot transfer, to test whether the
energy features drive cross-corpus transfer (cf. es-en CSW being quieter).

Trains BM CS-vs-mono logistic regression on: all 103 features, then dropping
F0 / energy / duration in turn, evaluating zero-shot on SEAME and MaSaC (raw
and per-speaker z-scored), five speaker-disjoint seeds.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "scripts")
import paper_feature_selection as pfs

spec = importlib.util.spec_from_file_location("clf44", "scripts/44_diagnostic_classifier.py")
clf44 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clf44)

SEEDS = [0, 1, 2, 3, 4]


def eval_subset(train_df, test_df, keep_feats, ztr, zte):
    sp_col = clf44.speaker_col(train_df)
    rows = []
    for seed in SEEDS:
        tr_m, _ = clf44.speaker_disjoint_split(train_df[sp_col].values, seed=seed)
        X_tr, y_tr, _, _ = clf44.load_xy(train_df.iloc[tr_m], zscore_within_speaker=ztr, feats=keep_feats)
        X_te, y_te, _, _ = clf44.load_xy(test_df, zscore_within_speaker=zte, feats=keep_feats)
        scaler = StandardScaler().fit(X_tr)
        clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced", random_state=seed)
        clf.fit(scaler.transform(X_tr), y_tr)
        prob = clf.predict_proba(scaler.transform(X_te))[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append(dict(seed=seed, auroc=roc_auc_score(y_te, prob),
                         accuracy=accuracy_score(y_te, pred),
                         f1=f1_score(y_te, pred, zero_division=0)))
    return pd.DataFrame(rows)


def main():
    bm = pd.read_csv("features/bangor_miami_disvoice_utt.csv", low_memory=False)
    seame = pd.read_csv("results/functional_anchors/seame/seame_integrated_features.csv", low_memory=False)
    masac = pd.read_csv("results/functional_anchors/masac/masac_integrated_features.csv", low_memory=False)

    common = [c for c in pfs.prosody_feature_columns(bm)
              if c in set(pfs.prosody_feature_columns(seame))
              and c in set(pfs.prosody_feature_columns(masac))]
    cat = {c: pfs.disvoice_category(c) for c in common}
    ablations = {
        "all": common,
        "drop_F0": [c for c in common if cat[c] != "f0"],
        "drop_energy": [c for c in common if cat[c] != "energy"],
        "drop_duration": [c for c in common if cat[c] != "duration"],
    }
    out = []
    for abl, feats in ablations.items():
        for tname, tdf in [("SEAME", seame), ("MaSaC", masac)]:
            for lab, (ztr, zte) in {"raw": (False, False), "z": (True, True)}.items():
                r = eval_subset(bm, tdf, feats, ztr, zte)
                out.append(dict(ablation=abl, nfeat=len(feats), test=tname, feat=lab,
                                auroc=r.auroc.mean(), auroc_sd=r.auroc.std(),
                                acc=r.accuracy.mean(), f1=r.f1.mean()))
                print(f"{abl:14s} ({len(feats):3d}f) {tname:6s} {lab:3s}: "
                      f"AUROC={r.auroc.mean():.3f}±{r.auroc.std():.3f} "
                      f"acc={r.accuracy.mean():.3f} f1={r.f1.mean():.3f}", flush=True)
    df = pd.DataFrame(out)
    df.to_csv("results/paper/diagnostic_classifier/ablation_by_category.csv", index=False)
    print("\nSaved -> results/paper/diagnostic_classifier/ablation_by_category.csv")


if __name__ == "__main__":
    main()
