"""
Build a deduplicated Bangor Miami utterance manifest from the segmented audio.

The corpus was delivered as audio only (no transcripts/labels). Segments are
named {conversation}_{start}_{end}.wav and duplicated across the four
`audio segments*` folders. We dedup by basename (keep first occurrence) and
parse conversation + start/end from the filename.

Label columns (lang, condition, speaker_id) are left EMPTY: they must be filled
from the Bangor CHAT (.cha) transcripts or the prior pipeline's labeled table
before the diagnostic classifier can train.
"""
from __future__ import annotations

import csv
import os
import pathlib
import re

BM_ROOT = pathlib.Path("data/bangor miami")
SEG_DIRS = ["audio segments", "audio segments 2", "audio segments 3", "audio segments 4"]
OUT = pathlib.Path("manifests/bangor_miami_manifest.csv")

pat = re.compile(r"^(?P<conv>.+?)_(?P<start>\d+(?:\.\d+)?)_(?P<end>\d+(?:\.\d+)?)\.wav$", re.I)


def main():
    seen = {}
    total = 0
    for d in SEG_DIRS:
        base = BM_ROOT / d
        if not base.exists():
            continue
        for conv_dir in os.scandir(base):
            if not conv_dir.is_dir():
                continue
            for e in os.scandir(conv_dir.path):
                if not e.name.lower().endswith(".wav"):
                    continue
                total += 1
                if e.name in seen:
                    continue
                m = pat.match(e.name)
                if not m:
                    continue
                conv = m.group("conv")
                start = float(m.group("start"))
                end = float(m.group("end"))
                utt_id = e.name[:-4]
                seen[e.name] = {
                    "utt_id": utt_id,
                    "wav": e.path,
                    "file_id": conv,
                    "start_time": start,
                    "end_time": end,
                    "speaker_id": "",   # TODO: fill from .cha speaker tier
                    "lang": "",         # TODO: fill from .cha language tags
                    "condition": "",    # TODO: code_switched / monolingual_EN / monolingual_ES
                }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(seen.values(), key=lambda r: (r["file_id"], r["start_time"]))
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    convs = sorted({r["file_id"] for r in rows})
    print(f"scanned {total} wavs across {len(SEG_DIRS)} folders")
    print(f"unique segments: {len(rows)}  across {len(convs)} conversations")
    print(f"conversations: {', '.join(convs)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
