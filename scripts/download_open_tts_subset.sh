#!/usr/bin/env bash
# scripts/download_open_tts_subset.sh
# Quick downloader for a fraction of the open_tts 5% sample archive.
# Requirements: aria2c (preferred) or wget, tar, shuf, python3, grep, head.
# Usage example:
#   bash scripts/download_open_tts_subset.sh --fraction 0.01 --outdir open_tts_1pct
# This will download the 5% sample tarball (if not cached) and extract ~1% of its audio files
# together with the metadata CSV into the specified directory.

set -euo pipefail

################################################################################
# Defaults
################################################################################
FRACTION="0.01"
OUTDIR="open_tts_subset"
# Official 5% sample archive published by the ishine/open_tts repository
URL_SAMPLE="https://ru-open-stt.ams3.cdn.digitaloceanspaces.com/radio_pspeech_sample_mp3.tar.gz"
CONNECTIONS=16

################################################################################
# Helpers
################################################################################
usage() {
  cat <<EOF
Usage: $0 [--fraction 0.01] [--outdir DIR]
  --fraction, -f   Fraction (0 < f <= 1) of files inside the 5% sample to extract. Default: 0.01
  --outdir, -o    Output directory. Will be created if missing. Default: open_tts_subset
  --help, -h      Show this help.
EOF
  exit 1
}

################################################################################
# Parse arguments
################################################################################
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fraction|-f)
      FRACTION="$2"; shift 2;;
    --outdir|-o)
      OUTDIR="$2"; shift 2;;
    --help|-h)
      usage;;
    *)
      echo "Unknown option: $1" >&2
      usage;;
  esac
done

# Validate fraction
export FRACTION
python - <<'PY'
import sys, os
f = float(os.environ["FRACTION"])
if not 0 < f <= 1:
    sys.exit("Fraction must be between 0 and 1")
PY

################################################################################
# Prepare working dirs
################################################################################
mkdir -p "$OUTDIR"
TEMP_DIR="$(mktemp -d)"
TARBALL="$TEMP_DIR/sample.tar.gz"

################################################################################
# Download sample archive (≈10 GB) if not already in temp
################################################################################
if [[ ! -f "$TARBALL" ]]; then
  echo ">> Downloading 5% sample archive…"
  if command -v aria2c &>/dev/null; then
    aria2c -x"$CONNECTIONS" -s"$CONNECTIONS" -o "$TARBALL" "$URL_SAMPLE"
  else
    echo "aria2c not found, falling back to wget (single stream, slower)" >&2
    wget -O "$TARBALL" "$URL_SAMPLE"
  fi
else
  echo ">> Reusing downloaded archive at $TARBALL"
fi

echo ">> Listing archive contents…"
# Generate complete list of files inside the archive
FILELIST="$TEMP_DIR/filelist.txt"
tar -tzf "$TARBALL" > "$FILELIST"

audio_list="$TEMP_DIR/audio_files.txt"
grep -E '\\.(mp3|wav)$' "$FILELIST" > "$audio_list"
TOTAL=$(wc -l < "$audio_list")
export TOTAL

if [[ "$TOTAL" -eq 0 ]]; then
  echo "Archive appears to contain no audio files. Exiting." >&2
  exit 2
fi

echo "Total audio files in sample archive: $TOTAL"

################################################################################
# Determine number of files to extract
################################################################################
WANTED=$(python - <<PY
import math, sys, os
print(max(1, math.floor(float(os.environ['TOTAL']) * float(os.environ['FRACTION']))))
PY
)

echo "Selecting $WANTED files (~$(awk "BEGIN {print $FRACTION*100}" )%)"

################################################################################
# Select random subset
################################################################################
shuf "$audio_list" | head -n "$WANTED" > "$TEMP_DIR/selected.txt"
# Also include CSV metadata files (lightweight)
grep -E '\\.(csv)$' "$FILELIST" >> "$TEMP_DIR/selected.txt"

################################################################################
# Extract selected files
################################################################################
EXTRACT_LIST="$TEMP_DIR/selected.txt"

echo ">> Extracting to $OUTDIR …"
# --no-same-owner avoids permission issues if running as root
# -T specifies file list

tar -xzf "$TARBALL" -C "$OUTDIR" --no-same-owner -T "$EXTRACT_LIST"

echo ">> Extraction finished. Cleaning up temp files…"
rm -rf "$TEMP_DIR"

echo "Done. Subset placed in $OUTDIR" 