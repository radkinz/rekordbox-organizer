# DJ Library Sorter

Automatically sorts audio files into genre folders using a trained
OpenL3 + scikit-learn classifier.

## Background
In my free time, I like to DJ, but I’ve always struggled with keeping my music library organized by genre. My initial goal was to use an existing music-genre classifier to automatically organize my Rekordbox library. However, I quickly realized that most available classifiers are trained on broad, conventional genres and don’t capture the more specific DJ-centric styles, particularly subgenres and variations of electronic music.

I decided to use this as an opportunity to build a custom pipeline from scratch. I curated my own dataset, trained a genre classifier tailored to DJ use cases, and then built the automated sorting tool contained in this repository.

The trained classifier is published separately on Hugging Face and can be found [here](https://huggingface.co/radkinz/dj-genre-openl3)


## What it does 
This script:
- scans an `unsorted/` folder
- classifies each audio file by genre
- copies or moves files into `sorted/<genre>/`
- shows a progress bar
- writes a CSV report of predictions

The classifier itself is published on Hugging Face.

## Supported Genres
- Jersey Club
- Dariacore
- House
- Techno
- Pop

## Requirements
- Python 3.8+ (recommended to create a conda environment)
- ffmpeg (recommended)
- See `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

## Usage (from repo root)
```bash
  conda activate openl3
  python scripts/sort_unsorted.py
```

## Move instead of copy (destructive)
```bash
  python scripts/sort_unsorted.py --move
```

## Recurse into subfolders
``` bash
  python scripts/sort_unsorted.py --recursive
```

## Custom folders
``` bash
  python scripts/sort_unsorted.py --in_dir ./unsorted --out_dir ./sorted
```
## Notes
- Files are always sorted using the highest-confidence predicted genre
- Confidence scores are recorded in the CSV report for inspection

## Next Steps
### Model Improvements
- Expanding the training dataset with additional DJ-centric genres (Dubstep, Trance, Hardgroove, Drum & Bass, Jungle, UK Garage, Baile Funk)
- Exploring transformer-based audio models (e.g., wav2vec2 / AST) as alternatives to OpenL3
- Extending the classifier to support multi-label outputs

### DJ Sorter
- Rekordbox-friendly playlist generation (.m3u8 exports by genre)
- Writing predicted genres directly to audio file metadata (ID3 tags)