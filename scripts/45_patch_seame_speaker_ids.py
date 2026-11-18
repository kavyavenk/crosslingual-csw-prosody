#!/usr/bin/env python3
"""Patch SEAME manifests with global speaker_id columns."""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from seame_speaker_utils import attach_global_speaker_columns


def patch_manifest(path: pathlib.Path) -> None:
    df = pd.read_csv(path, low_memory=False)
    df = attach_global_speaker_columns(df)
    df.to_csv(path, index=False)
    print(
        f"{path}: {len(df)} utts, "
        f"{df['speaker_id'].nunique()} global speakers, "
        f"{df['speaker_raw'].nunique()} legacy ints"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "manifests",
        nargs="*",
        default=[
            "manifests/seame_manifest.csv",
            "manifests/seame_manifest_audio.csv",
        ],
    )
    args = p.parse_args()
    for m in args.manifests:
        patch_manifest(pathlib.Path(m))


if __name__ == "__main__":
    main()
