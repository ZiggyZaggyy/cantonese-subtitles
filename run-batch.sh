#!/usr/bin/env bash
# Batch transcribe 七女性 → Cantonese SRT (Cantonese-Whisper-Medium).
#
# Setup:
#   /usr/local/opt/python@3.12/bin/python3.12 -m venv ~/.venvs/cantonese-asr
#   source ~/.venvs/cantonese-asr/bin/activate
#   pip install -r requirements.txt
#
# Smoke test (90s dialog from ep 5):
#   ./run-batch.sh test
#
# Full batch (7 episodes, ~1–2 h/episode speech time on Intel CPU; uses VAD):
#   ./run-batch.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-$HOME/.venvs/cantonese-asr}"
VIDEO_DIR="${VIDEO_DIR:-$HOME/Downloads/bilibili-videos}"
OUT_DIR="${OUT_DIR:-$VIDEO_DIR/subtitles-yue-whisper}"
MODEL="${MODEL:-reachan/Cantonese-Whisper-Medium}"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Create venv first — see header in this script."
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

PY="$ROOT/transcribe_cantonese.py"

if [[ "${1:-}" == "test" ]]; then
  EP5=( "$VIDEO_DIR"/*第五集*.mp4 )
  exec python -u "$PY" --model "$MODEL" --start-seconds 300 --test-seconds 90 --out-dir /tmp/yue-asr-test "${EP5[0]}"
fi

FILES=()
while IFS= read -r f; do
  FILES+=("$f")
done < <(find "$VIDEO_DIR" -maxdepth 1 -name '*.mp4' -type f)

SORTED=()
while IFS= read -r f; do
  SORTED+=("$f")
done < <(python3 - <<'PY' "${FILES[@]}"
import re, sys
from pathlib import Path

def ep_num(name: str) -> int:
    for pat, n in [
        (r"第一集", 1), (r"第二集", 2), (r"第三集", 3), (r"第四集", 4),
        (r"第五集", 5), (r"第六集", 6), (r"完结篇", 7),
    ]:
        if re.search(pat, name):
            return n
    return 99

for p in sorted(sys.argv[1:], key=lambda p: ep_num(Path(p).name)):
    print(p)
PY
)
FILES=("${SORTED[@]}")
TO_RUN=()
for f in "${FILES[@]}"; do
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f" 2>/dev/null || echo 0)
  ok=$(awk -v d="$dur" 'BEGIN { exit !(d >= 2700) }' && echo yes || echo no)
  [[ "$ok" == yes ]] || { echo "SKIP (too short): $(basename "$f")"; continue; }
  TO_RUN+=("$f")
done

if [[ ${#TO_RUN[@]} -eq 0 ]]; then
  echo "No complete episodes found in $VIDEO_DIR"
  exit 1
fi

echo "Transcribing ${#TO_RUN[@]} files → $OUT_DIR (model: $MODEL)"
exec python -u "$PY" --model "$MODEL" --skip-existing --out-dir "$OUT_DIR" "${TO_RUN[@]}"
