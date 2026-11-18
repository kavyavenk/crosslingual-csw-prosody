"""
Extract 103 DisVoice prosody features for Bangor Miami utterance segments.

Reads manifests/bangor_miami_manifest.csv (built by 51_build_bm_manifest.py) and
writes features/bangor_miami_disvoice_utt.csv, carrying through the metadata
columns (utt_id, wav, file_id, start_time, end_time, speaker_id, lang,
condition). Label columns (speaker_id/lang/condition) pass through as-is, so
this can run BEFORE labels are joined; the diagnostic classifier only needs them
populated at train time.

Resumable: skips utt_ids already present in the output file.

Usage:
  python3 scripts/52_extract_bm_disvoice.py [--limit N]
"""
from __future__ import annotations

import argparse
import os
import pathlib

import pandas as pd
import tqdm

import fix_disvoice_compatibility  # noqa: F401  (patches scipy/urllib before disvoice import)
from disvoice.prosody.prosody import Prosody

MAN = pathlib.Path("manifests/bangor_miami_manifest.csv")
OUT = pathlib.Path("features/bangor_miami_disvoice_utt.csv")
META = ["utt_id", "wav", "file_id", "start_time", "end_time", "speaker_id", "lang", "condition"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="extract only first N (debug)")
    ap.add_argument("--save-every", type=int, default=500)
    args = ap.parse_args()

    if not MAN.exists():
        raise SystemExit(f"Manifest not found: {MAN} (run 51_build_bm_manifest.py)")
    df = pd.read_csv(MAN)
    if args.limit:
        df = df.head(args.limit)

    done = set()
    rows = []
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = set(prev["utt_id"].astype(str))
        rows = prev.to_dict("records")

    pros = Prosody()
    n_new = 0
    for r in tqdm.tqdm(df.itertuples(), total=len(df)):
        if str(r.utt_id) in done:
            continue
        wav = str(r.wav)
        if not os.path.exists(wav):
            continue
        try:
            fdf = pros.extract_features_file(wav, static=True, plots=False, fmt="dataframe")
        except Exception as e:
            print("skip", wav, e)
            continue
        if fdf is None or len(fdf) == 0:
            continue
        row = {m: getattr(r, m, "") for m in META}
        row.update({k: float(v) for k, v in fdf.iloc[0].items()})
        rows.append(row)
        n_new += 1
        if n_new % args.save_every == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False)

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[bangor_miami] wrote {OUT} with {len(rows)} rows (+{n_new} new)")


if __name__ == "__main__":
    main()
