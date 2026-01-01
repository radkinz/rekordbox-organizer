"""
Sort files from ./unsorted into ./sorted/<genre>/ using a trained OpenL3+sklearn classifier.

What it does
- Scans ./unsorted (optionally recursive) for audio files
- For each file: MP3/audio -> OpenL3 embedding -> classifier -> predicted genre
- Copies (default) or moves files into ./sorted/<genre>/
- Writes a CSV report: ./data/predictions/unsorted_predictions.csv
- Shows a tqdm progress bar

If mp3 decoding fails for some files, you can point to a working ffmpeg:
  setx FFMPEG_EXE "C:\\ffmpeg\\bin\\ffmpeg.exe"
Then reopen terminal and rerun.

Notes
- No confidence barrier: always uses the top predicted label.
- Confidence is still recorded in the report CSV when available (predict_proba).
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Tuple

import joblib
import librosa
import numpy as np
import openl3
from tqdm import tqdm

SR = 48000
SEGMENT_SECONDS = 20
CONTENT_TYPE = "music"
INPUT_REPR = "mel256"
EMBEDDING_SIZE = 512
HOP_SIZE = 1.0
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}

def sanitize_folder_name(name: str) -> str:
    bad = '<>:"/\\|?*'
    for ch in bad:
        name = name.replace(ch, "_")
    name = name.strip()
    return name if name else "Unknown"

def unique_dest_path(dest_dir: Path, filename: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if not dest.exists():
        return dest

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    i = 2
    while True:
        cand = dest_dir / f"{stem} ({i}){suffix}"
        if not cand.exists():
            return cand
        i += 1

def ffmpeg_available(ffmpeg_exe: str) -> bool:
    try:
        subprocess.run([ffmpeg_exe, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def try_librosa_load(path: Path) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Try loading with librosa (soundfile -> audioread). Returns (audio, error_str).
    """
    try:
        y, _ = librosa.load(str(path), sr=SR, mono=True)
        return y.astype(np.float32, copy=False), None
    except Exception as e:
        return None, repr(e)

def load_with_ffmpeg(path: Path, ffmpeg_exe: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Decode via ffmpeg -> temp wav -> librosa. Returns (audio, error_str).
    """
    tmp_wav = Path(tempfile.gettempdir()) / f"{path.stem}_{uuid.uuid4().hex}.wav"
    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(path),
        "-ac", "1",
        "-ar", str(SR),
        str(tmp_wav),
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        y, _ = librosa.load(str(tmp_wav), sr=SR, mono=True)
        return y.astype(np.float32, copy=False), None
    except Exception as e:
        return None, f"ffmpeg decode failed: {repr(e)}"
    finally:
        try:
            if tmp_wav.exists():
                tmp_wav.unlink()
        except Exception:
            pass

def embed_audio(y: np.ndarray) -> np.ndarray:
    """
    Compute a single (512,) OpenL3 embedding for a clip (pads/trims to fixed length).
    """
    seg_len = int(SEGMENT_SECONDS * SR)

    # fixed-size input reduces TF retracing + makes behavior consistent
    if len(y) < seg_len:
        y = np.pad(y, (0, seg_len - len(y)))
    else:
        y = y[:seg_len]

    emb, _ = openl3.get_audio_embedding(
        y,
        SR,
        content_type=CONTENT_TYPE,
        input_repr=INPUT_REPR,
        embedding_size=EMBEDDING_SIZE,
        hop_size=HOP_SIZE,
        verbose=False,
    )
    return emb.mean(axis=0).astype(np.float32, copy=False)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="./unsorted", help="Folder containing new files to sort")
    ap.add_argument("--out_dir", default="./sorted", help="Output sorted folder")
    ap.add_argument("--model", default="./models/dj_genre_openl3/genre_clf.joblib", help="Trained classifier .joblib")
    ap.add_argument("--report", default="./data/predictions/unsorted_predictions.csv", help="CSV report output")
    ap.add_argument("--move", action="store_true", help="Move files instead of copy (destructive)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders of in_dir")
    ap.add_argument(
        "--ffmpeg",
        default=os.environ.get("FFMPEG_EXE", "ffmpeg"),
        help="ffmpeg executable name/path (env var FFMPEG_EXE recommended for absolute path)",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    model_path = Path(args.model)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {in_dir.resolve()}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.resolve()}")

    clf = joblib.load(model_path)
    has_proba = hasattr(clf, "predict_proba")

    use_ffmpeg = ffmpeg_available(args.ffmpeg)
    if not use_ffmpeg:
        print(f"[WARN] ffmpeg not usable as '{args.ffmpeg}'. Will rely on librosa only.")
        print("       If some files fail to decode, set FFMPEG_EXE to a working ffmpeg.exe path.")

    # Collect files
    if args.recursive:
        files = [p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    else:
        files = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS]

    print(f"Found {len(files)} audio file(s) in {in_dir.resolve()}")

    n_sorted = 0
    n_failed = 0
    n_skipped_non_audio = 0  # kept for possible future expansion

    with report_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "source_path",
                "dest_path",
                "predicted_label",
                "confidence",
                "status",
                "error",
            ],
        )
        w.writeheader()

        for src in tqdm(files, desc="Sorting tracks", unit="file", dynamic_ncols=True):
            # Decode audio
            y, err = try_librosa_load(src)
            if y is None and use_ffmpeg:
                y, err = load_with_ffmpeg(src, args.ffmpeg)

            if y is None:
                n_failed += 1
                w.writerow(
                    {
                        "source_path": str(src),
                        "dest_path": "",
                        "predicted_label": "",
                        "confidence": "",
                        "status": "DECODE_FAILED",
                        "error": err or "unknown decode error",
                    }
                )
                continue

            # Embed + predict
            try:
                v = embed_audio(y)
                X = v.reshape(1, -1)
                pred = clf.predict(X)[0]

                conf = ""
                if has_proba:
                    probs = clf.predict_proba(X)[0]
                    conf = float(np.max(probs))

                label_folder = sanitize_folder_name(str(pred))
                dest_dir = out_dir / label_folder
                dest = unique_dest_path(dest_dir, src.name)

                if args.move:
                    shutil.move(str(src), str(dest))
                else:
                    shutil.copy2(str(src), str(dest))

                n_sorted += 1
                w.writerow(
                    {
                        "source_path": str(src),
                        "dest_path": str(dest),
                        "predicted_label": str(pred),
                        "confidence": conf if conf != "" else "",
                        "status": "OK",
                        "error": "",
                    }
                )

            except Exception as e:
                n_failed += 1
                w.writerow(
                    {
                        "source_path": str(src),
                        "dest_path": "",
                        "predicted_label": "",
                        "confidence": "",
                        "status": "PREDICT_FAILED",
                        "error": repr(e),
                    }
                )

    print("\nDone.")
    print(f"  sorted: {n_sorted}")
    print(f"  failed: {n_failed}")
    print(f"  report: {report_path.resolve()}")
    print(f"  out_dir: {out_dir.resolve()}")
    if args.move:
        print("  NOTE: --move was used (original files were moved out of unsorted).")
    else:
        print("  NOTE: default COPY used (unsorted remains unchanged).")


if __name__ == "__main__":
    main()
