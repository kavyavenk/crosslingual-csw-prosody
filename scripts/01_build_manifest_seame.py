#!/usr/bin/env python3
"""
Build SEAME manifest(s) for prosody analysis.

Reads the Phase II annotation CSV and writes:
- Full manifest (all annotated utterances)
- Audio manifest (utterances with a resolvable local WAV clip)
"""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from seame_audio_index import build_seame_audio_index, resolve_seame_audio
from seame_speaker_utils import global_speaker_key


def build_seame_manifest(
    csv_file,
    output_manifest,
    audio_manifest=None,
    fallback_wav_dir="data/SEAME/interview_aligned/interview_aligned",
):
    print(f"Reading SEAME data from: {csv_file}")
    df = pd.read_csv(csv_file, low_memory=False)
    print(f"Total rows in CSV: {len(df)}")

    required_cols = ["file", "start_time", "end_time", "speaker", "language"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        return None

    df["start_sec"] = df["start_time"] / 1000.0
    df["end_sec"] = df["end_time"] / 1000.0
    df["duration_sec"] = df["end_sec"] - df["start_sec"]

    lang_map = {"CS": "CS", "EN": "EN", "ZH": "ZH"}
    df["lang"] = df["language"].map(lang_map).fillna("UNK")
    df["condition"] = df["lang"].map(
        {
            "CS": "code_switched",
            "EN": "monolingual_EN",
            "ZH": "monolingual_ZH",
        }
    ).fillna("unknown")
    df = df[df["lang"] != "UNK"].copy()

    transcript_col = (
        "filtered_transcript_2015"
        if "filtered_transcript_2015" in df.columns
        else "transcript_2015"
    )
    df["text"] = df[transcript_col].astype(str)
    df["file_id"] = df["file"].astype(str)
    df["speaker_raw"] = df["speaker"]
    df["speaker_id"] = [
        global_speaker_key(fid, raw) for fid, raw in zip(df["file_id"], df["speaker_raw"])
    ]
    df["speaker"] = df["speaker_id"]
    df["utt_id"] = (
        df["file_id"]
        + "_"
        + df["start_sec"].map(lambda x: f"{x:.3f}")
        + "_"
        + df["end_sec"].map(lambda x: f"{x:.3f}")
    )

    print("Building SEAME audio index (one pass over WAV roots)...")
    audio_index = build_seame_audio_index()
    print(f"  Indexed {len(audio_index):,} clip keys")

    fallback_root = pathlib.Path(fallback_wav_dir)

    def resolve_wav(row):
        hit = resolve_seame_audio(
            row["file_id"],
            row["start_sec"],
            row["end_sec"],
            int(row["start_time"]),
            int(row["end_time"]),
            audio_index,
        )
        if hit:
            return hit
        start_str = f"{row['start_sec']:.3f}".rstrip("0").rstrip(".")
        end_str = f"{row['end_sec']:.3f}".rstrip("0").rstrip(".")
        exact = fallback_root / f"{row['file_id']}_{start_str}_{end_str}.wav"
        if exact.exists():
            return str(exact)
        ms = fallback_root / f"{row['file_id']}_{int(row['start_time'])}_{int(row['end_time'])}.wav"
        if ms.exists():
            return str(ms)
        return str(exact)

    print("Resolving WAV paths...")
    df["wav"] = df.apply(resolve_wav, axis=1)
    df["wav_resolved"] = df["wav"].map(lambda p: pathlib.Path(p).exists())

    keep = [
        "utt_id",
        "file_id",
        "wav",
        "start",
        "end",
        "speaker",
        "speaker_id",
        "speaker_raw",
        "lang",
        "condition",
        "text",
        "duration_sec",
    ]
    df = df.rename(columns={"start_sec": "start", "end_sec": "end"})
    manifest_df = df[keep].copy()

    output_manifest = pathlib.Path(output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(output_manifest, index=False)

    print(f"\nWrote manifest to: {output_manifest}")
    print(f"  - Total utterances: {len(manifest_df):,}")
    print(f"  - Global speaker IDs: {manifest_df['speaker_id'].nunique()}")
    print(f"  - WAV resolved on disk: {df['wav_resolved'].sum():,}")

    if audio_manifest:
        audio_df = manifest_df[df["wav_resolved"].values].copy()
        audio_path = pathlib.Path(audio_manifest)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_df.to_csv(audio_path, index=False)
        print(f"\nWrote audio manifest to: {audio_path}")
        print(f"  - Utterances with audio: {len(audio_df):,}")
        print(f"  - Speakers: {audio_df['speaker_id'].nunique()}")

    print("\nLanguage distribution:")
    for lang, count in manifest_df["lang"].value_counts().items():
        print(f"  {lang}: {count:,} ({100 * count / len(manifest_df):.1f}%)")

    return manifest_df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build SEAME manifest for prosody analysis")
    p.add_argument(
        "--csv",
        default="data/SEAME/SEAME_data_annotation_new_2015_annotated_4_17_24.csv",
    )
    p.add_argument("--output", default="manifests/seame_manifest.csv")
    p.add_argument(
        "--audio-output",
        default="manifests/seame_manifest_audio.csv",
        help="Subset with resolvable local WAV clips (for DisVoice extraction)",
    )
    args = p.parse_args()

    csv_file = pathlib.Path(args.csv)
    if not csv_file.exists():
        print(f"ERROR: CSV file not found: {csv_file}")
        sys.exit(1)

    build_seame_manifest(
        csv_file,
        args.output,
        audio_manifest=args.audio_output,
    )
