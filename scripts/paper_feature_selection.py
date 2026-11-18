"""
Unified outcome selection for the CSW prosody paper.

Rule (Tier 1 — single rule everywhere):
  Within each prosodic category (F0, duration, energy), rank numeric DisVoice
  features by between-condition separation variance:
      Var_c( mean(feature | condition=c) )
  over c in {code_switched, monolingual_EN, monolingual_L1}.
  Take the top k_per_category (default 3) per category → 9 primary outcomes.

Used for: Table 4 (MeM), §5.2 OLS count summaries, §5.3 interlocutor models,
functional-anchor stratified t-tests (same 9 features, not F0-only).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

CONDITION_CS = "code_switched"
CONDITION_EN = "monolingual_EN"
L1_CONDITIONS = {"monolingual_ZH", "monolingual_HI"}


def disvoice_category(feature: str) -> str:
    f = feature.lower()
    # Energy first: any feature computed over the energy contour of voiced/
    # unvoiced segments is an energy feature, INCLUDING energy tilt/mse
    # (e.g. avgtiltEvoiced, stdmseEunvoiced). These must be caught before the
    # "tilt" -> F0 rule so energy-contour tilt is not misfiled as F0.
    if "energy" in f or "intensity" in f or "evoiced" in f or "eunvoiced" in f:
        return "energy"
    # F0 / pitch, including F0 tilt (F0tilt*) and F0 mse (F0mse*).
    if f[0].isdigit() or "f0" in f or "tilt" in f:
        return "f0"
    if any(x in f for x in ("dur", "pause", "rate", "silence")):
        return "duration"
    return "other"


def prosody_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {
        "utt_id",
        "speaker",
        "speaker_id",
        "speaker_raw",
        "lang",
        "condition",
        "file_id",
        "start",
        "end",
        "start_time",
        "end_time",
        "duration_sec",
        "text",
        "wav",
        "n_switches",
        "n_discourse_markers",
        "n_repetitions",
        "m_index",
        "i_index",
        "is_l1_to_l2",
        "is_csw",
        "has_discourse_marker",
        "has_repetition",
        "gender",
        "age_bucket",
        "nationality",
        "gender_match",
        "interlocutor_gender",
        "prev_speaker",
        "interaction_id",
    }
    cols = []
    for c in df.columns:
        if c in exclude or c.startswith("Unnamed"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def condition_separation_variance(
    df: pd.DataFrame, feature: str, condition_col: str = "condition"
) -> float:
    sub = df[[condition_col, feature]].dropna()
    if sub.empty:
        return 0.0
    means = sub.groupby(condition_col)[feature].mean()
    if len(means) < 2:
        return 0.0
    return float(means.var())


def select_primary_outcomes(
    df: pd.DataFrame,
    k_per_category: int = 3,
    categories: Sequence[str] = ("f0", "duration", "energy"),
    condition_col: str = "condition",
) -> Dict[str, List[str]]:
    """Return {category: [feature, ...]} length k_per_category each."""
    feats = prosody_feature_columns(df)
    scored: Dict[str, List[tuple]] = {c: [] for c in categories}
    for f in feats:
        cat = disvoice_category(f)
        if cat not in scored:
            continue
        v = condition_separation_variance(df, f, condition_col)
        scored[cat].append((f, v))
    out: Dict[str, List[str]] = {}
    for cat in categories:
        ranked = sorted(scored[cat], key=lambda x: x[1], reverse=True)
        out[cat] = [f for f, _ in ranked[:k_per_category]]
    return out


def primary_outcome_list(
    df: pd.DataFrame, k_per_category: int = 3, **kwargs
) -> List[str]:
    d = select_primary_outcomes(df, k_per_category=k_per_category, **kwargs)
    ordered: List[str] = []
    for cat in ("f0", "duration", "energy"):
        for f in d.get(cat, []):
            if f not in ordered:
                ordered.append(f)
    return ordered
