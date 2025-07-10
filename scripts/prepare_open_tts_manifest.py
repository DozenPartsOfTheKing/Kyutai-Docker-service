#!/usr/bin/env python3
"""Prepare Kyutai-compatible manifests from the Open-TTS/ Open-STT folder structure.

Features
--------
1. Recursively scans a dataset directory for audio files (.opus, .mp3, .wav).
2. For каждого аудио ищет соседний TXT с тем же базовым именем.
3. Конвертирует звук в mono 24 kHz WAV (ffmpeg) — можно изменить через --sr.
4. Записывает `train.jsonl` и `val.jsonl` (формат Kyutai).

Usage
-----
python scripts/prepare_open_tts_manifest.py \
       --datadir  /path/to/open_tts_full \
       --outdir   /path/to/tts_wav24k     \
       --workers  16                      \
       --split    0.9                     \
       --sr       24000                   

Designed to be idempotent: если WAV уже существует — не перекодирует.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

audio_exts = {".wav", ".mp3", ".opus"}


def find_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """Return list of (audio_path, txt_path)."""
    pairs = []
    for ext in audio_exts:
        for audio in root.rglob(f"*{ext}"):
            txt = audio.with_suffix(".txt")
            if txt.exists():
                pairs.append((audio, txt))
    return pairs


def convert_to_wav(args: Tuple[Path, Path, Path, int, bool]):
    src_audio, dst_audio, sr, keep_original = args  # type: ignore
    if dst_audio.exists():
        return dst_audio
    dst_audio.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-loglevel",
        "quiet",
        "-i",
        str(src_audio),
        "-ar",
        str(sr),
        "-ac",
        "1",
        str(dst_audio),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] ffmpeg failed on {src_audio}: {e}", file=sys.stderr)
        return None
    if not keep_original:
        try:
            src_audio.unlink()
        except Exception:
            pass
    return dst_audio


def main():
    parser = argparse.ArgumentParser(description="Prepare Kyutai TTS manifests from Open-TTS dataset")
    parser.add_argument("--datadir", required=True, type=Path, help="Root folder with *.opus/*.txt pairs")
    parser.add_argument("--outdir", required=True, type=Path, help="Where to store converted WAVs & manifests")
    parser.add_argument("--split", type=float, default=0.9, help="Train split fraction (default 0.9)")
    parser.add_argument("--sr", type=int, default=24000, help="Target sample rate for WAV")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8, help="Parallel ffmpeg processes")
    parser.add_argument("--limit_hours", type=float, help="Optional cap on total hours to process")
    parser.add_argument("--keep_original", action="store_true", help="Do NOT delete source audio after conversion")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    wav_dir = args.outdir / "wav"
    wav_dir.mkdir(exist_ok=True)

    print("[1/4] Scanning dataset…", flush=True)
    pairs = find_pairs(args.datadir)
    if not pairs:
        print("No audio+txt pairs found", file=sys.stderr)
        sys.exit(1)

    if args.limit_hours is not None:
        # Rough subsample by random shuffle (assuming avg 4s → 900 files/hour)
        random.shuffle(pairs)
        approx_needed = int(args.limit_hours * 900)
        pairs = pairs[:approx_needed]

    print(f"   found {len(pairs):,} audio/text pairs")

    # Prepare conversion tasks
    tasks = []
    for src_audio, _ in pairs:
        dst_audio = wav_dir / (src_audio.stem + ".wav")
        tasks.append((src_audio, dst_audio, args.sr, args.keep_original))

    print("[2/4] Converting to WAV (ffmpeg)…")
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(convert_to_wav, tasks))

    # Filter out failed conversions
    good_pairs = []
    for (src_audio, txt), dst_audio in zip(pairs, results):
        if dst_audio is None or not dst_audio.exists():
            continue
        good_pairs.append((dst_audio, txt))

    print(f"   OK {len(good_pairs):,} files")

    print("[3/4] Writing manifests…")
    random.shuffle(good_pairs)
    split_idx = int(len(good_pairs) * args.split)
    train_pairs = good_pairs[:split_idx]
    val_pairs = good_pairs[split_idx:]

    def write_manifest(pairs: Iterable[Tuple[Path, Path]], fp: Path):
        with fp.open("w", encoding="utf-8") as f:
            for wav, txt in pairs:
                text = txt.read_text(encoding="utf-8", errors="ignore").strip()
                record = {
                    "audio": str(wav.resolve()),
                    "text": text,
                    "speaker_id": 0,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    train_fp = args.outdir / "train.jsonl"
    val_fp = args.outdir / "val.jsonl"
    write_manifest(train_pairs, train_fp)
    write_manifest(val_pairs, val_fp)

    print("[4/4] Done!")
    print(f"   Train: {train_fp}  ({len(train_pairs):,} lines)")
    print(f"   Val:   {val_fp}    ({len(val_pairs):,} lines)")


if __name__ == "__main__":
    main() 