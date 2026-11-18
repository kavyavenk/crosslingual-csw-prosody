#!/usr/bin/env python3
"""
Apply the FINAL MaSaC annotation sheet as the single source of truth.

Source of truth:
  "to publish final with automated" sheet of
  `MaSac transcripts with strategy annotations (1).xlsx`

This propagates the final LID labels, LID_tags, and CS-strategy annotations into
the three files the analysis pipeline reads, so that all downstream counts,
directionality/M-I indices, strategy splits, and mixed-effects coefficients are
recomputed from the corrected data:

  1. data/masac_raw/masac_data_compiled.csv        (Language, LID_tags -> manifest lang/condition + directionality)
  2. data/masac_raw/fixed_with_automated_annotations.csv (Type of code-switching -> cs_strategy)
  3. manifests/masac_manifest.csv                  (lang, condition, text)

Join key: sheet `name` (e.g. "test_100_0.wav") -> file_id "test_100_0".
The MaSaC manifest uses utt_id == file_id and keeps audio timing untouched, so we
patch labels in place instead of rebuilding (no librosa / audio needed).
"""

import pathlib
import shutil
import sys

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
SHEET_XLSX = (
    pathlib.Path.home() / "Downloads" / "MaSac transcripts with strategy annotations (1).xlsx"
)
SHEET_NAME = "to publish final with automated"

COMPILED = REPO / "data/masac_raw/masac_data_compiled.csv"
ANNOTATIONS = REPO / "data/masac_raw/fixed_with_automated_annotations.csv"
MANIFEST = REPO / "data/../manifests/masac_manifest.csv"
FEATURES = REPO / "features/masac_disvoice_utt.csv"
FINAL_COPY = REPO / "data/masac_raw/masac_final_publish_sheet.csv"

LANG_MAP = {"CSW": "CS", "EN": "EN", "HI": "HI"}
COND_MAP = {"CS": "code_switched", "EN": "monolingual_EN", "HI": "monolingual_HI"}


def _fid(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(".wav", "", regex=False)


def load_sheet() -> pd.DataFrame:
    if not SHEET_XLSX.exists():
        sys.exit(f"ERROR: source sheet not found: {SHEET_XLSX}")
    df = pd.read_excel(SHEET_XLSX, sheet_name=SHEET_NAME)
    df["file_id"] = _fid(df["name"])
    if df["file_id"].duplicated().any():
        sys.exit("ERROR: duplicate file_id in final sheet")
    return df


def backup(path: pathlib.Path):
    bak = path.with_suffix(path.suffix + ".prefinal.bak")
    if path.exists() and not bak.exists():
        shutil.copy2(path, bak)
        print(f"  backup -> {bak.name}")


def patch_compiled(sheet: pd.DataFrame):
    print(f"[1/3] {COMPILED.relative_to(REPO)}")
    df = pd.read_csv(COMPILED, low_memory=False)
    df["file_id"] = _fid(df["name"])
    look = sheet.set_index("file_id")
    df["Language"] = df["file_id"].map(look["Language from LID tool"]).fillna(df["Language"])
    df["LID_tags"] = df["file_id"].map(look["LID_tags"]).fillna(df["LID_tags"])
    df["Transcript"] = df["file_id"].map(look["Transcript"]).fillna(df["Transcript"])
    df = df.drop(columns=["file_id"])
    backup(COMPILED)
    df.to_csv(COMPILED, index=False)
    print("  Language:", dict(df["Language"].value_counts()))


def patch_annotations(sheet: pd.DataFrame):
    print(f"[2/3] {ANNOTATIONS.relative_to(REPO)}")
    out = sheet.copy()
    # Match legacy column name expected elsewhere.
    if "Automatic English Translation" in out.columns:
        out = out.rename(columns={"Automatic English Translation": "English Translation"})
    out.insert(0, "Unnamed: 0", range(len(out)))
    out = out.drop(columns=["file_id"], errors="ignore")
    backup(ANNOTATIONS)
    out.to_csv(ANNOTATIONS, index=False)
    print("  Type of code-switching:", dict(out["Type of code-switching"].astype(str).value_counts().head(6)))


def patch_manifest(sheet: pd.DataFrame):
    print(f"[3/3] {MANIFEST.resolve().relative_to(REPO)}")
    df = pd.read_csv(MANIFEST, low_memory=False)
    look = sheet.set_index("file_id")
    new_lang = df["file_id"].map(look["Language from LID tool"]).map(LANG_MAP)
    missing = new_lang.isna().sum()
    if missing:
        print(f"  WARNING: {missing} manifest rows had no sheet match (kept old label)")
    df["lang"] = new_lang.fillna(df["lang"])
    df["condition"] = df["lang"].map(COND_MAP).fillna(df["condition"])
    df["text"] = df["file_id"].map(look["Transcript"]).fillna(df["text"])
    backup(MANIFEST)
    df.to_csv(MANIFEST, index=False)
    print("  condition:", dict(df["condition"].value_counts()))


def patch_features(sheet: pd.DataFrame):
    """Refresh the DisVoice feature file's cached lang/condition (keyed by utt_id==file_id)."""
    if not FEATURES.exists():
        print(f"[4/4] SKIP (missing): {FEATURES.relative_to(REPO)}")
        return
    print(f"[4/4] {FEATURES.relative_to(REPO)}")
    df = pd.read_csv(FEATURES, low_memory=False)
    if "condition" not in df.columns or "utt_id" not in df.columns:
        print("  no condition/utt_id columns; skipping")
        return
    look = sheet.set_index("file_id")
    new_lang = df["utt_id"].astype(str).map(look["Language from LID tool"]).map(LANG_MAP)
    if "lang" in df.columns:
        df["lang"] = new_lang.fillna(df["lang"])
    df["condition"] = new_lang.map(COND_MAP).fillna(df["condition"])
    backup(FEATURES)
    df.to_csv(FEATURES, index=False)
    print("  condition:", dict(df["condition"].value_counts()))


def main():
    sheet = load_sheet()
    print(f"Final sheet rows: {len(sheet)}")
    sheet.to_csv(FINAL_COPY, index=False)
    print(f"Saved provenance copy -> {FINAL_COPY.relative_to(REPO)}\n")
    patch_compiled(sheet)
    patch_annotations(sheet)
    patch_manifest(sheet)
    patch_features(sheet)
    print("\nDone. Re-run the MaSaC pipeline to refresh derived features.")


if __name__ == "__main__":
    main()
