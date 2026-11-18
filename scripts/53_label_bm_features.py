"""
Join Bangor Miami annotations onto the extracted DisVoice feature table.

Matches on (File, Start timestamp, End timestamp) rounded to 3 decimals, which is
an exact 1:1 match with the segment filenames. Populates:
  condition   : 'code_switched' if is_code_switched else 'monolingual'
  speaker_id  : '<File>_<Speaker>'  (globally unique for speaker-disjoint splits)
  lang        : Lang code from the sheet
  gender      : Speaker gender
plus CS-property columns (direction, cs_type, n_switches, intersentential, dominance).
"""
from __future__ import annotations

import pathlib
import pandas as pd

FEAT = pathlib.Path("features/bangor_miami_disvoice_utt.csv")
XLSX = pathlib.Path.home() / "Downloads" / "full_final_miami_bangor_annotated.xlsx"
SHEET = "with-span-ann"
ND = 3


def main():
    feat = pd.read_csv(FEAT)
    ann = pd.read_excel(XLSX, sheet_name=SHEET)

    ann = ann.copy()
    ann["_k"] = list(zip(ann["File"], ann["Start timestamp"].round(ND), ann["End timestamp"].round(ND)))
    ann = ann.drop_duplicates("_k")
    amap = ann.set_index("_k")

    feat["_k"] = list(zip(feat["file_id"], feat["start_time"].round(ND), feat["end_time"].round(ND)))
    matched = feat["_k"].isin(amap.index)
    print(f"feature rows: {len(feat)} | matched to annotation: {int(matched.sum())}")

    def g(col):
        return feat["_k"].map(amap[col])

    is_cs = g("is_code_switched")
    feat["condition"] = is_cs.map(lambda v: "code_switched" if v is True or v == True else "monolingual")
    feat["speaker_id"] = g("File").astype(str) + "_" + g("Speaker").astype(str)
    feat["speaker"] = feat["speaker_id"]
    feat["lang"] = g("Lang")
    feat["gender"] = g("Speaker gender")
    feat["direction"] = g("Direction of code-switches")
    feat["cs_type"] = g("Type of code-switching")
    feat["n_switches"] = g("Number of code-switches")
    feat["intersentential"] = g("Intersentential code-switch?")
    feat["dominance"] = g("dominance")

    feat = feat.drop(columns=["_k"])
    feat.to_csv(FEAT, index=False)

    print("condition:\n", feat["condition"].value_counts(dropna=False).to_string())
    print("n speakers:", feat["speaker_id"].nunique())
    print("wrote", FEAT)


if __name__ == "__main__":
    main()
