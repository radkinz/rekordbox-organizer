#!/usr/bin/env python3
"""
Download the trained model artifacts from Hugging Face into ./models/.

Usage:
  python download_model.py --repo radkinz/dj-genre-openl3
  python download_model.py --repo radkinz/dj-genre-openl3 --revision v1.0
  python download_model.py --repo radkinz/dj-genre-openl3 --out_dir models

This downloads:
  - genre_clf.joblib
  - label_map.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


FILES = ["genre_clf.joblib", "label_map.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Hugging Face repo id, e.g. radkinz/dj-genre-openl3")
    ap.add_argument("--revision", default=None, help="Optional git tag/branch/commit, e.g. v1.0")
    ap.add_argument("--out_dir", default="./models", help="Where to place downloaded files")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from: {args.repo}" + (f" @ {args.revision}" if args.revision else ""))
    for fname in FILES:
        local_path = hf_hub_download(
            repo_id=args.repo,
            filename=fname,
            revision=args.revision,
        )
        dest = out_dir / fname
        dest.write_bytes(Path(local_path).read_bytes())
        print(f"  ✓ {fname} -> {dest.resolve()}")

    print("\nDone. Your model files are ready in ./models/.")


if __name__ == "__main__":
    main()
