"""Global SEAME speaker IDs matching LDC speaker-info.xls registry."""

from __future__ import annotations

import re
from typing import Optional, Union


def global_speaker_key(file_id: str, speaker_raw: Optional[Union[int, str]] = None) -> str:
    """
    Map a SEAME file_id (+ annotation speaker int) to a corpus-wide speaker key.

    Keys align with LDC ``speaker-info.xls``:
    - Conversation* (61 speakers): ``Conversation{N}`` from annotation speaker int
    - Interview-Malaysia* (29) / Interview-Singapore* (66): from filename + country code

    Conversation filenames share a session prefix (e.g. ``01`` in ``01NC01FBX_0101``),
    so the prefix alone is not a speaker ID. Use ``speaker_raw`` when available.
    """
    fid = str(file_id)
    try:
        int(fid[0:2])
        if speaker_raw is not None and str(speaker_raw).strip() not in ("", "nan"):
            return f"Conversation{int(speaker_raw)}"
        m = re.match(r"^\d{2}NC(\d{2})", fid)
        if m:
            return f"Conversation{int(m.group(1))}"
        return f"Conversation{int(fid[0:2])}"
    except ValueError:
        speaker_id = int(fid[2:4])
        country = "Malaysia" if fid[5] == "A" else "Singapore"
        return f"Interview-{country}{speaker_id}"


def attach_global_speaker_columns(df, file_col: str = "file_id", speaker_col: str = "speaker"):
    """Add speaker_raw (annotation integer) and speaker_id (global key); set speaker to global."""
    out = df.copy()
    if "speaker_raw" not in out.columns:
        out["speaker_raw"] = out[speaker_col]
    out["speaker_id"] = [
        global_speaker_key(fid, raw)
        for fid, raw in zip(out[file_col], out["speaker_raw"])
    ]
    out["speaker"] = out["speaker_id"]
    return out
