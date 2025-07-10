#!/usr/bin/env bash
# scripts/finetune_tts_pipeline.sh
# One-click pipeline: convert OpenTTS dataset ➜ prepare manifests ➜ launch QLoRA fine-tuning.
#
# Usage:
#   bash scripts/finetune_tts_pipeline.sh \
#        --datadir  /data/open_tts_full \
#        --checkpoint_dir /checkpoints/tts_ru \
#        --epochs 3
#
# Requirements: ffmpeg, python deps (see requirements.txt), torch+cuda.
set -euo pipefail

################################################################################
# Defaults
################################################################################
DATA_DIR=""
CKPT_DIR="./checkpoints/tts_ru"
EPOCHS=3
SPLIT=0.9
SR=24000
WORKERS=$(nproc)
BATCH_SIZE=4
GRAD_ACCUM=4
LIMIT_HOURS=""

################################################################################
usage() {
  cat <<EOF
finetune_tts_pipeline.sh - end-to-end Kyutai TTS QLoRA fine-tune

Arguments:
  --datadir DIR          Root with OpenTTS audio+txt (required)
  --checkpoint_dir DIR   Where to save checkpoints (default ./checkpoints/tts_ru)
  --epochs N             Training epochs (default 3)
  --split FLOAT          Train split fraction (default 0.9)
  --sr INT               Target sample rate (default 24000)
  --workers N            Parallel ffmpeg workers (default: nproc)
  --help                 Show this help
EOF
  exit 1
}

################################################################################
# Parse CLI
################################################################################
while [[ $# -gt 0 ]]; do
  case "$1" in
    --datadir)
      DATA_DIR="$2"; shift 2;;
    --checkpoint_dir)
      CKPT_DIR="$2"; shift 2;;
    --epochs)
      EPOCHS="$2"; shift 2;;
    --split)
      SPLIT="$2"; shift 2;;
    --sr)
      SR="$2"; shift 2;;
    --workers)
      WORKERS="$2"; shift 2;;
    --limit_hours)
      LIMIT_HOURS="$2"; shift 2;;
    --help|-h)
      usage;;
    *)
      echo "Unknown option $1"; usage;;
  esac
done

[[ -z "$DATA_DIR" ]] && { echo "--datadir is required"; usage; }

################################################################################
# Paths
################################################################################
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
OUT_DIR="$DATA_DIR/_wav${SR}"
TRAIN_MANIFEST="$OUT_DIR/train.jsonl"
VAL_MANIFEST="$OUT_DIR/val.jsonl"
CONFIG_FILE="$CKPT_DIR/config.toml"

###############################################################################
# Step 1: Convert & build manifests (idempotent)
###############################################################################
if [[ ! -f "$TRAIN_MANIFEST" ]]; then
  echo "[PIPE] Preparing manifests …"
  python "$SCRIPT_DIR/prepare_open_tts_manifest.py" \
         --datadir "$DATA_DIR" \
         --outdir  "$OUT_DIR" \
         --workers "$WORKERS" \
         --split   "$SPLIT" \
         --sr      "$SR" \
         ${LIMIT_HOURS:+--limit_hours $LIMIT_HOURS}
else
  echo "[PIPE] Reusing existing manifests at $OUT_DIR"
fi

###############################################################################
# Step 2: Create training config
###############################################################################
mkdir -p "$CKPT_DIR"
cat > "$CONFIG_FILE" <<EOT
[data]
manifest     = "$TRAIN_MANIFEST"
val_manifest = "$VAL_MANIFEST"

[tokenizer]
path = "hf://kyutai/tts-1.6b-en_fr/tokenizer_spm_8k_en_fr_audio.model"
kind = "phoneme"

[lora]
r      = 16
alpha  = 32
dropout = 0.1

[train]
batch_size = ${BATCH_SIZE}
grad_accum = ${GRAD_ACCUM}
epochs      = ${EPOCHS}
precision   = "fp16"
output_dir  = "$CKPT_DIR"
EOT

echo "[PIPE] Config written to $CONFIG_FILE"

###############################################################################
# Step 3: Launch training (torchrun 2 GPUs if available)
###############################################################################
NUM_GPU=$(python - <<'PY'
import torch, os; print(torch.cuda.device_count())
PY
)

if [[ "$NUM_GPU" -gt 1 ]]; then
  echo "[PIPE] Launching multi-GPU training on $NUM_GPU GPUs …"
  torchrun --nproc_per_node=$NUM_GPU \
    moshi-train --config "$CONFIG_FILE"
else
  echo "[PIPE] Launching single GPU training …"
  moshi-train --config "$CONFIG_FILE"
fi

###############################################################################
# Step 4: Quick inference test (optional)
###############################################################################
TEST_WAV="$CKPT_DIR/test_hello.wav"
echo "Привет, мир!" | python "$SCRIPT_DIR/test_tts_qlora.py" \
       --checkpoint "$CKPT_DIR" \
       --out "$TEST_WAV"

echo "[PIPE] Pipeline finished! Sample synthesized audio → $TEST_WAV" 