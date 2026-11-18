import argparse, pandas as pd, pathlib, tqdm, shutil
from disvoice.prosody.prosody import Prosody

p = argparse.ArgumentParser()
p.add_argument("--corpus", choices=["masac","seame"], required=True)
args = p.parse_args()

MAN = pathlib.Path(f"manifests/{args.corpus}_manifest.csv")
CLIPS = pathlib.Path(f"data/{args.corpus}_wav16k_clips")
OUT = pathlib.Path(f"features/{args.corpus}_disvoice_utt.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

if not MAN.exists():
    raise SystemExit(f"Manifest not found: {MAN}")

pros = Prosody()

df = pd.read_csv(MAN)
feats = []
for r in tqdm.tqdm(df.itertuples(), total=len(df)):
    # Get WAV file path from manifest
    wav_file = pathlib.Path(r.wav) if hasattr(r, 'wav') and r.wav else None
    
    # Try the path from manifest first
    audio_path = None
    if wav_file and wav_file.exists():
        audio_path = str(wav_file)
    else:
        # Fallback: try clips directory with utt_id
        clip = (CLIPS / f"{r.utt_id}.wav")
        if clip.exists():
            audio_path = str(clip)
        # For MASAC, also try file_id directly
        elif args.corpus == "masac" and hasattr(r, 'file_id'):
            # MASAC files are in subdirectories, try to find them
            file_id = r.file_id
            possible_paths = [
                CLIPS / f"{file_id}.wav",
                CLIPS / "test" / "audios" / f"{file_id}.wav",
                CLIPS / "train" / "audios" / f"{file_id}.wav",
                CLIPS / "val" / "audios" / f"{file_id}.wav",
            ]
            for p in possible_paths:
                if p.exists():
                    audio_path = str(p)
                    break
    
    if not audio_path:
        continue
    
    try:
        fdict = pros.extract_features_file(audio_path)
        row = {"utt_id": r.utt_id, "speaker": r.speaker, "lang": r.lang, "condition": r.condition}
        row.update({k: float(v) for k,v in fdict.items()})
        feats.append(row)
    except Exception as e:
        print("skip", audio_path, e)

pd.DataFrame(feats).to_csv(OUT, index=False)
print(f"[{args.corpus}] Wrote {OUT} with {len(feats)} rows")
