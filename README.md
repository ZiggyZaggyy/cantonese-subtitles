# cantonese-subtitles

Cantonese TV → SRT pipeline: Silero VAD, Whisper fine-tunes (`reachan/Cantonese-Whisper-Medium`, optional `awong-dev/whisper-large-v3-yue-lora-dec-enc4`), and optional SenseVoice-Small ONNX.

Originally developed alongside [ZiggyZaggyy/sofa-salon](https://github.com/ZiggyZaggyy/sofa-salon); maintained as a standalone tool.

## Setup

```bash
python3.12 -m venv ~/.venvs/cantonese-asr
source ~/.venvs/cantonese-asr/bin/activate
pip install -r requirements.txt
```

Optional SenseVoice path:

```bash
pip install -r requirements-sensevoice.txt
```

Requires **ffmpeg** and **ffprobe** on `PATH`.

## Quick start

Batch all episodes in a video folder (default: `~/Downloads/bilibili-videos`):

```bash
./run-batch.sh
```

Ep1 iterative runs (opening 5 min smoke tests, v9/v10 A/B):

```bash
./iterate-ep1.sh open-5min-v9
./iterate-ep1.sh open-5min-v9-sensevoice
./iterate-ep1.sh compare-v9-v10
```

Translate Cantonese SRT → Mandarin (needs `OPENAI_API_KEY`):

```bash
python translate_srt.py path/to/out.srt -o path/to/out-zh.srt
```

## Environment

| Variable | Default |
|----------|---------|
| `VENV` | `~/.venvs/cantonese-asr` |
| `VIDEO_DIR` | `~/Downloads/bilibili-videos` |
| `OUT_DIR` | `$VIDEO_DIR/subtitles-yue-whisper` |
| `MODEL` | `reachan/Cantonese-Whisper-Medium` |

## Scripts

| File | Purpose |
|------|---------|
| `transcribe_cantonese.py` | Main Whisper + VAD → SRT |
| `transcribe_sensevoice.py` | SenseVoice-Small ONNX (same VAD pipeline) |
| `iterate-ep1.sh` | Ep1 5 min / full runs and compares |
| `run-batch.sh` | Batch all episodes |
| `compare_srt.py` | Diff two SRT files |
| `inspect_alignment.py` | Cue timing vs audio energy |
| `translate_srt.py` | Cantonese → Mandarin via OpenAI |
