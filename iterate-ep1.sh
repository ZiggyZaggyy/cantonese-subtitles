#!/usr/bin/env bash
# Iterative ep1 transcription: VAD output alongside original, with compare/merge.
# Phase 1 — opening 5 min (validate before full episode):
#   ./iterate-ep1.sh open-5min
#
# Phase 2 — full episode (after open-5min looks good):
#   ./iterate-ep1.sh full
#
# Compare only:
#   ./iterate-ep1.sh compare
#
# Optional Mandarin subs (needs OPENAI_API_KEY — ChatGPT-quality, not automatic):
#   python translate_srt.py path/to/out.srt -o path/to/out-zh.srt

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-$HOME/.venvs/cantonese-asr}"
VIDEO_DIR="${VIDEO_DIR:-$HOME/Downloads/bilibili-videos}"
LEGACY_DIR="$VIDEO_DIR/subtitles-yue-whisper/ep01"
VAD_BASE="$VIDEO_DIR/subtitles-yue-whisper/ep01-vad"
PY="$ROOT/transcribe_cantonese.py"
SENSEVOICE_PY="$ROOT/transcribe_sensevoice.py"
CMP="$ROOT/compare_srt.py"
MEDIUM_MODEL="reachan/Cantonese-Whisper-Medium"
MEDIUM_PROCESSOR="openai/whisper-medium"
LARGE_MODEL="awong-dev/whisper-large-v3-yue-lora-dec-enc4"
LARGE_PROCESSOR="openai/whisper-large-v3"

EP1=( "$VIDEO_DIR"/*第一集*.mp4 )

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Create venv first — see run-batch.sh header."
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

run_vad() {
  local label="$1"
  local seconds="$2"
  shift 2
  local model="${ASR_MODEL:-$MEDIUM_MODEL}"
  local processor="${ASR_PROCESSOR:-$MEDIUM_PROCESSOR}"
  local out="$VAD_BASE/$label"
  mkdir -p "$out"
  echo "=== VAD transcribe → $out (${seconds}s, ${model}) ==="
  python -u "$PY" \
    --model "$model" \
    --processor "$processor" \
    --vad-threshold 0.35 \
    --speech-pad-ms 250 \
    --out-dir "$out" \
    --no-episode-subdir \
    --test-seconds "$seconds" \
    "$@" \
    "${EP1[0]}"
  python "$CMP" "$out/out.srt" "$LEGACY_DIR/out.srt" \
    --from-sec 0 --to-sec "$seconds" \
    --report-out "$out/compare.txt" \
    --merge-out "$out/out-merged.srt"
  echo "VAD:    $out/out.srt"
  echo "Merged: $out/out-merged.srt"
  echo "Report: $out/compare.txt"
}

run_sensevoice_v9() {
  local seconds="$1"
  local out="$VAD_BASE/open-5min-v9"
  local py="$VENV/bin/python"
  mkdir -p "$out"
  if ! "$py" -c "from funasr_onnx.sensevoice_bin import SenseVoiceSmall" 2>/dev/null; then
    echo "Installing SenseVoice ONNX deps…"
    "$py" -m pip install -r "$ROOT/requirements-sensevoice.txt"
  fi
  echo "=== SenseVoice-Small (v9 VAD) → $out/out-sensevoice.srt (${seconds}s) ==="
  "$py" -u "$SENSEVOICE_PY" \
    --vad-threshold 0.35 \
    --speech-pad-ms 250 \
    --out-dir "$out" \
    --output-stem out-sensevoice \
    --no-episode-subdir \
    --test-seconds "$seconds" \
    --no-short-gap-vad \
    "${EP1[0]}"
  if [[ -f "$out/out.srt" ]]; then
    "$py" "$CMP" "$out/out-sensevoice.srt" "$out/out.srt" \
      --from-sec 0 --to-sec "$seconds" \
      --report-out "$out/compare-sensevoice-vs-v9.txt" \
      --merge-out "$out/out-sensevoice-merged-with-v9.srt"
    echo "Compare vs v9: $out/compare-sensevoice-vs-v9.txt"
  else
    echo "No v9 out.srt yet — run open-5min-v9 first for side-by-side compare."
  fi
  echo "SenseVoice: $out/out-sensevoice.srt"
}

compare_v9_v10() {
  local v9="$VAD_BASE/open-5min-v9/out.srt"
  local v10="$VAD_BASE/open-5min-v10/out.srt"
  local report="$VAD_BASE/open-5min-v10/compare-v9-v10.txt"
  if [[ ! -f "$v9" || ! -f "$v10" ]]; then
    echo "Need both $v9 and $v10 — run open-5min-v9 and open-5min first."
    exit 1
  fi
  python - "$v9" "$v10" "$report" "$ROOT" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[4])
from compare_srt import parse_srt, report

v9, v10, out = map(Path, sys.argv[1:4])
text = report("v9", parse_srt(v9), "v10", parse_srt(v10), t0=0.0, t1=300.0)
out.write_text(text, encoding="utf-8")
print(text)
print(f"\nReport → {out}", file=sys.stderr)
PY
}

case "${1:-}" in
  open-5min)
    run_vad "open-5min-v10" 300
    ;;
  open-5min-v9)
    run_vad "open-5min-v9" 300 --no-short-gap-vad
    ;;
  open-5min-v9-large)
    ASR_MODEL="$LARGE_MODEL" ASR_PROCESSOR="$LARGE_PROCESSOR" \
      run_vad "open-5min-v9-large" 300 --no-short-gap-vad
    ;;
  open-5min-v9-sensevoice)
    run_sensevoice_v9 300
    ;;
  compare-v9-v10)
    compare_v9_v10
    ;;
  full)
    out="$VAD_BASE/full"
    mkdir -p "$out"
    echo "=== Full ep1 VAD → $out ==="
    python -u "$PY" \
      --model reachan/Cantonese-Whisper-Medium \
      --vad-threshold 0.35 \
      --speech-pad-ms 250 \
      --cue-shift-sec 0.30 \
      --out-dir "$out" \
      --no-episode-subdir \
      "${EP1[0]}"
    python "$CMP" "$out/out.srt" "$LEGACY_DIR/out.srt" \
      --report-out "$out/compare.txt" \
      --merge-out "$out/out-merged.srt"
    ;;
  compare)
    python "$CMP" "$VAD_BASE/open-5min/out.srt" "$LEGACY_DIR/out.srt" \
      --from-sec 60 --to-sec 300
    ;;
  *)
    echo "Usage: $0 {open-5min|open-5min-v9|open-5min-v9-large|open-5min-v9-sensevoice|compare-v9-v10|full|compare}"
    exit 1
    ;;
esac
