"""Build a lookup index for SEAME utterance-level WAV clips across local roots."""

from __future__ import annotations

import pathlib
import re
from typing import Dict, Optional, Tuple

Key = Tuple[str, float, float]

SEAME_AUDIO_ROOTS = (
    pathlib.Path("data/seame_wav16k_clips"),
    pathlib.Path("data/SEAME/interview_aligned/interview_aligned"),
    pathlib.Path("data/SEAME more"),
)


def _keys_from_stem(stem: str) -> Optional[list[Key]]:
    stem = re.sub(r"_(mod\d+|result)$", "", stem)
    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        return None
    file_id, start_s, end_s = parts
    try:
        s_f, e_f = float(start_s), float(end_s)
    except ValueError:
        return None
    return [
        (file_id, round(s_f, 3), round(e_f, 3)),
        (file_id, int(round(s_f * 1000)), int(round(e_f * 1000))),
    ]


def build_seame_audio_index(roots=SEAME_AUDIO_ROOTS) -> Dict[Key, str]:
    index: Dict[Key, str] = {}
    for root in roots:
        if not root.exists():
            continue
        for wav in root.rglob("*.wav"):
            keys = _keys_from_stem(wav.stem)
            if not keys:
                continue
            for key in keys:
                index.setdefault(key, str(wav))
    return index


def resolve_seame_audio(
    file_id: str,
    start_sec: float,
    end_sec: float,
    start_ms: int,
    end_ms: int,
    index: Dict[Key, str],
) -> Optional[str]:
    for key in (
        (file_id, round(start_sec, 3), round(end_sec, 3)),
        (file_id, start_ms, end_ms),
    ):
        hit = index.get(key)
        if hit:
            return hit
    return None
