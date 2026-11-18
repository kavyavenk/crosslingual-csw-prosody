"""Full diagnostic-classifier ablation grid (BM / SEAME / MaSaC × feature subsets)."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_feature_selection as pfs

SEEDS = [0, 1, 2, 3, 4]
ROOT = pathlib.Path(__file__).resolve().parents[1]


def speaker_disjoint_split(sp, test_size=0.2, seed=0):
    uniq = np.unique(sp)
    tr, te = train_test_split(uniq, test_size=test_size, random_state=seed)
    return np.isin(sp, tr), np.isin(sp, te)


def canonical_103() -> list[str]:
    return pd.read_csv(ROOT / "features/first_contrast_seame_CS_vs_EN.csv")["feature"].tolist()


def load_xy(df, zscore_within_speaker=False, feats=None):
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
    return X.fillna(0.0).values, y, sp, feats


def feats_by_category(all_feats, drop_cats=(), keep_cats=None):
    groups = {"f0": [], "duration": [], "energy": [], "other": []}
    for f in all_feats:
        groups[pfs.disvoice_category(f)].append(f)
    if keep_cats is not None:
        return [f for cat in keep_cats for f in groups[cat]]
    keep = []
    for cat, fl in groups.items():
        if cat not in drop_cats:
            keep.extend(fl)
    return keep


FEATURE_CONFIGS = [
    ("all103", dict()),
    ("no_energy", dict(drop_cats=("energy",))),
    ("no_duration", dict(drop_cats=("duration",))),
    ("no_f0", dict(drop_cats=("f0",))),
    ("no_duration_energy", dict(drop_cats=("duration", "energy"))),
    ("f0_only", dict(keep_cats=("f0",))),
    ("duration_only", dict(keep_cats=("duration",))),
    ("energy_only", dict(keep_cats=("energy",))),
]


def resolve_feats(base_feats, cfg):
    if "keep_cats" in cfg:
        return feats_by_category(base_feats, keep_cats=cfg["keep_cats"])
    return feats_by_category(base_feats, drop_cats=cfg.get("drop_cats", ()))


def eval_train_test(train_df, test_df, feats, z_tr, z_te, eval_label):
    if not feats:
        return []
    sp_col = "speaker_id" if "speaker_id" in train_df.columns else "speaker"
    rows = []
    for seed in SEEDS:
        tr_m, _ = speaker_disjoint_split(train_df[sp_col].values, seed=seed)
        _, te_hold = speaker_disjoint_split(train_df[sp_col].values, seed=seed)
        X_tr, y_tr, _, _ = load_xy(train_df.iloc[tr_m], zscore_within_speaker=z_tr, feats=feats)
        X_eval, y_eval, _, _ = load_xy(test_df, zscore_within_speaker=z_te, feats=feats)
        if len(np.unique(y_eval)) < 2:
            continue
        scaler = StandardScaler().fit(X_tr)
        clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced", random_state=seed)
        clf.fit(scaler.transform(X_tr), y_tr)
        prob = clf.predict_proba(scaler.transform(X_eval))[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append(
            dict(
                seed=seed,
                eval=eval_label,
                auroc=float(roc_auc_score(y_eval, prob)),
                accuracy=float(accuracy_score(y_eval, pred)),
                n_feat=len(feats),
            )
        )
    return rows


def summarize(rows, eval_label, meta):
    if not rows:
        return None
    d = pd.DataFrame(rows)
    d = d[d["eval"] == eval_label]
    if d.empty:
        return None
    out = {**meta, "eval": eval_label, "n_feat": int(d["n_feat"].iloc[0])}
    for m in ["auroc", "accuracy"]:
        out[m] = float(d[m].mean())
        out[f"{m}_std"] = float(d[m].std())
    return out


def main():
    bm = pd.read_csv(ROOT / "features/bangor_miami_disvoice_utt.csv", low_memory=False)
    se = pd.read_csv(
        ROOT / "results/functional_anchors/seame/seame_integrated_features.csv",
        low_memory=False,
    )
    ma = pd.read_csv(
        ROOT / "results/functional_anchors/masac/masac_integrated_features.csv",
        low_memory=False,
    )
    corpora = {"BM": bm, "SEAME": se, "MaSaC": ma}
    base = [f for f in canonical_103() if all(f in df.columns for df in corpora.values())]

    results = []
    for train_name, train_df in corpora.items():
        others = {k: v for k, v in corpora.items() if k != train_name}
        for label, cfg in FEATURE_CONFIGS:
            feats = resolve_feats(base, cfg)
            for z_tr, z_te, norm in [(False, False, "raw"), (True, True, "z")]:
                sp_col = "speaker_id" if "speaker_id" in train_df.columns else "speaker"
                hold_rows = []
                for seed in SEEDS:
                    tr_m, _ = speaker_disjoint_split(train_df[sp_col].values, seed=seed)
                    _, te_hold = speaker_disjoint_split(train_df[sp_col].values, seed=seed)
                    X_tr, y_tr, _, _ = load_xy(
                        train_df.iloc[tr_m], zscore_within_speaker=z_tr, feats=feats
                    )
                    X_ho, y_ho, _, _ = load_xy(
                        train_df.iloc[te_hold], zscore_within_speaker=z_tr, feats=feats
                    )
                    if len(feats) == 0 or len(np.unique(y_ho)) < 2:
                        continue
                    scaler = StandardScaler().fit(X_tr)
                    clf = LogisticRegression(
                        max_iter=5000, C=1.0, class_weight="balanced", random_state=seed
                    )
                    clf.fit(scaler.transform(X_tr), y_tr)
                    prob = clf.predict_proba(scaler.transform(X_ho))[:, 1]
                    pred = (prob >= 0.5).astype(int)
                    hold_rows.append(
                        dict(
                            seed=seed,
                            eval="holdout",
                            auroc=float(roc_auc_score(y_ho, prob)),
                            accuracy=float(accuracy_score(y_ho, pred)),
                            n_feat=len(feats),
                        )
                    )
                meta = dict(
                    train_corpus=train_name,
                    test_corpus=train_name,
                    feature_set=label,
                    norm=norm,
                    transfer="in_corpus",
                )
                s = summarize(hold_rows, "holdout", meta)
                if s:
                    results.append(s)

                for test_name, test_df in others.items():
                    zs_rows = []
                    for seed in SEEDS:
                        tr_m, _ = speaker_disjoint_split(train_df[sp_col].values, seed=seed)
                        X_tr, y_tr, _, _ = load_xy(
                            train_df.iloc[tr_m], zscore_within_speaker=z_tr, feats=feats
                        )
                        X_te, y_te, _, _ = load_xy(
                            test_df, zscore_within_speaker=z_te, feats=feats
                        )
                        if len(feats) == 0 or len(np.unique(y_te)) < 2:
                            continue
                        scaler = StandardScaler().fit(X_tr)
                        clf = LogisticRegression(
                            max_iter=5000, C=1.0, class_weight="balanced", random_state=seed
                        )
                        clf.fit(scaler.transform(X_tr), y_tr)
                        prob = clf.predict_proba(scaler.transform(X_te))[:, 1]
                        pred = (prob >= 0.5).astype(int)
                        zs_rows.append(
                            dict(
                                seed=seed,
                                eval="zero_shot",
                                auroc=float(roc_auc_score(y_te, prob)),
                                accuracy=float(accuracy_score(y_te, pred)),
                                n_feat=len(feats),
                            )
                        )
                    meta = dict(
                        train_corpus=train_name,
                        test_corpus=test_name,
                        feature_set=label,
                        norm=norm,
                        transfer="zero_shot",
                    )
                    s = summarize(zs_rows, "zero_shot", meta)
                    if s:
                        results.append(s)

    out_dir = ROOT / "results/paper/diagnostic_classifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "ablation_full_grid.csv", index=False)
    print(f"Wrote {len(df)} rows -> {out_dir / 'ablation_full_grid.csv'}")

    # compact pivot for main-table feature set only
    main = df[df.feature_set == "all103"].copy()
    main.to_csv(out_dir / "ablation_all103_by_corpus.csv", index=False)
    print(main.sort_values(["train_corpus", "test_corpus", "norm", "transfer"]).to_string())


if __name__ == "__main__":
    main()
