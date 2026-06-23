#!/usr/bin/env python3
"""Cantonese TV → SRT via FunAudioLLM/SenseVoiceSmall (free local ASR).

Reuses the v9 Silero VAD pipeline from transcribe_cantonese.py (per-region ASR,
gap fill, refine_chunks) but swaps Whisper for SenseVoice-Small with language=yue.

Install extra deps once (ONNX path — no full funasr / llvmlite build):
  pip install -r requirements-sensevoice.txt

First run downloads ~230 MB ONNX weights from HuggingFace into ~/.cache/sensevoice-small-onnx/.

5-min ep1 smoke test (writes alongside v9 whisper output):
  ./iterate-ep1.sh open-5min-v9-sensevoice
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
# Shared VAD / SRT pipeline (v9-compatible defaults).
from transcribe_cantonese import (  # noqa: E402
    DEFAULT_CUE_SHIFT_SEC,
    MIN_GAP_FILL_SEC,
    SAMPLE_RATE,
    GAP_FILL_THRESHOLD,
    _write_alignment_reports,
    episode_num,
    extract_wav,
    load_audio,
    load_vad,
    transcribe_with_vad,
    verify_result,
    write_srt,
)

DEFAULT_MODEL = "sensevoice-small-onnx"
ONNX_REPO = "DennisHuang648/SenseVoiceSmall-onnx"
BPE_REPO = "FunAudioLLM/SenseVoiceSmall"
BPE_FILE = "chn_jpn_yue_eng_ko_spectok.bpe.model"
CACHE_DIR = Path.home() / ".cache" / "sensevoice-small-onnx"
SENSEVOICE_TAG_RE = re.compile(r"<\|[^|]+\|>")
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
# SenseVoice-Small ONNX is much faster than Whisper-medium on CPU.
CPU_MINUTES_PER_AUDIO_MINUTE = 0.8


def normalize_sensevoice_text(raw: str) -> str:
    """Strip SenseVoice metadata tokens and match Whisper-style compact Cantonese."""
    if not raw:
        return ""
    text = SENSEVOICE_TAG_RE.sub("", raw).strip()
    if not CJK_RE.search(text):
        return ""
    return re.sub(r"\s+", "", text)


def ensure_sensevoice_onnx_model(model: str) -> Path:
    """Resolve local ONNX model dir; download from HuggingFace on first use."""
    model_path = Path(model).expanduser()
    if model_path.is_dir() and (model_path / "model_quant.onnx").exists():
        return model_path

    cache = CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    if not (cache / "model_quant.onnx").exists():
        from huggingface_hub import snapshot_download

        print(f"  downloading SenseVoice ONNX → {cache}…", flush=True)
        snapshot_download(ONNX_REPO, local_dir=str(cache))

    bpe = cache / BPE_FILE
    if not bpe.exists():
        from huggingface_hub import hf_hub_download

        print(f"  downloading SenseVoice tokenizer → {bpe}…", flush=True)
        hf_hub_download(BPE_REPO, BPE_FILE, local_dir=str(cache))

    return cache


class SenseVoiceASR:
    """Whisper-pipeline-compatible wrapper around SenseVoice-Small ONNX."""

    def __init__(self, model_dir: str, device: str = "cpu") -> None:
        from funasr_onnx.sensevoice_bin import SenseVoiceSmall

        del device  # ONNX runtime uses CPU (device_id=-1).
        path = ensure_sensevoice_onnx_model(model_dir)
        self.model_dir = str(path)
        self.model = SenseVoiceSmall(
            self.model_dir,
            quantize=True,
            batch_size=1,
            device_id="-1",
        )

    def __call__(
        self,
        inputs: dict,
        *,
        return_timestamps: bool = True,
        generate_kwargs: dict | None = None,
    ) -> dict:
        del generate_kwargs
        audio = np.asarray(inputs["array"], dtype=np.float32)
        sr = int(inputs["sampling_rate"])
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {sr}")
        seg_dur = len(audio) / sr
        if seg_dur < 0.05:
            return {"text": "", "chunks": []}

        results = self.model(audio, language="yue", textnorm="woitn")
        raw = (results[0] if results else "").strip()
        text = normalize_sensevoice_text(raw)
        if not text:
            return {"text": "", "chunks": []}

        chunks: list[dict] = []
        if return_timestamps:
            chunks = [{"text": text, "timestamp": (0.0, seg_dur)}]
        return {"text": text, "chunks": chunks}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cantonese SenseVoice-Small → SRT (v9 VAD pipeline)",
    )
    parser.add_argument("inputs", nargs="+", help="Video or WAV files")
    parser.add_argument(
        "--out-dir",
        default=str(Path.home() / "Downloads/bilibili-videos/subtitles-yue-whisper"),
    )
    parser.add_argument(
        "--output-stem",
        default="out-sensevoice",
        help="Base name for outputs (e.g. out-sensevoice → out-sensevoice.srt)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--start-seconds", type=float, default=0)
    parser.add_argument("--test-seconds", type=float, default=0, help="Only transcribe first N seconds")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-vad", action="store_true", help="Transcribe full audio (no Silero VAD)")
    parser.add_argument("--no-episode-subdir", action="store_true")
    parser.add_argument("--vad-threshold", type=float, default=0.35)
    parser.add_argument("--speech-pad-ms", type=int, default=250)
    parser.add_argument("--no-gap-fill", action="store_true")
    parser.add_argument(
        "--no-short-gap-vad",
        action="store_true",
        help="Skip isolated VAD on short inter-segment gaps (v9 behaviour)",
    )
    parser.add_argument("--min-gap-fill-sec", type=float, default=MIN_GAP_FILL_SEC)
    parser.add_argument("--gap-fill-threshold", type=float, default=GAP_FILL_THRESHOLD)
    parser.add_argument(
        "--cue-shift-sec",
        type=float,
        default=DEFAULT_CUE_SHIFT_SEC,
        help="Delay all cues by N seconds",
    )
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem

    print(f"Loading SenseVoice {args.model} on {args.device}…", flush=True)
    try:
        asr = SenseVoiceASR(args.model, device=args.device)
    except ImportError as exc:
        print(
            "SenseVoice ONNX deps missing "
            f"({exc}). Run:\n"
            "  pip install -r requirements-sensevoice.txt",
            file=sys.stderr,
        )
        return 1

    vad_model = get_speech_timestamps = None
    if not args.no_vad:
        print("Loading Silero VAD…", flush=True)
        vad_model, get_speech_timestamps = load_vad()

    failed: list[str] = []
    for inp in args.inputs:
        src = Path(inp)
        ep = episode_num(src.name)
        ep_dir = out_root if args.no_episode_subdir else (out_root / f"ep{ep:02d}" if ep else out_root / src.stem)
        ep_dir.mkdir(parents=True, exist_ok=True)
        srt_path = ep_dir / f"{stem}.srt"
        txt_path = ep_dir / f"{stem}.txt"

        if args.skip_existing and srt_path.exists() and srt_path.stat().st_size > 500:
            print(f"SKIP existing {srt_path}")
            continue

        print(f"\n=== {src.name} → {ep_dir} ({stem}) ===", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            if src.suffix.lower() in {".wav", ".flac"}:
                wav = src
            else:
                extract_wav(
                    src,
                    wav,
                    start_sec=args.start_seconds,
                    duration_sec=args.test_seconds or None,
                )
            audio, duration_sec = load_audio(wav)

        est_min = duration_sec / 60 * CPU_MINUTES_PER_AUDIO_MINUTE
        print(
            f"  audio {duration_sec / 60:.1f} min — SenseVoice transcribe (CPU ~{est_min:.1f} min)…",
            flush=True,
        )

        vad_stats: dict = {}
        if vad_model is not None and get_speech_timestamps is not None:
            text, chunks, vad_stats = transcribe_with_vad(
                asr,
                audio,
                vad_model,
                get_speech_timestamps,
                vad_threshold=args.vad_threshold,
                speech_pad_ms=args.speech_pad_ms,
                gap_fill=not args.no_gap_fill,
                min_gap_fill_sec=args.min_gap_fill_sec,
                gap_fill_threshold=args.gap_fill_threshold,
                short_gap_vad=not args.no_short_gap_vad,
                cue_shift_sec=args.cue_shift_sec,
            )
        else:
            result = asr(
                {"array": audio, "sampling_rate": SAMPLE_RATE},
                return_timestamps=True,
            )
            text = result.get("text", "")
            chunks = list(result.get("chunks") or [])
            if not chunks and text:
                chunks = [{"timestamp": (0.0, duration_sec), "text": text}]

        if not verify_result(text, duration_sec, vad_stats.get("speech_sec")):
            print(f"  VERIFY FAILED ({len(text)} chars)", flush=True)
            failed.append(src.name)
            continue

        from inspect_alignment import EP1_OPEN_CHECKPOINTS

        checkpoints = EP1_OPEN_CHECKPOINTS if duration_sec <= 320 else []
        pre_srt = ep_dir / f"{stem}-pre-align.srt"
        write_srt(chunks, pre_srt)
        pre_report, pre_warnings = _write_alignment_reports(
            audio,
            pre_srt,
            ep_dir,
            duration_sec=duration_sec,
            report_name=f"alignment-{stem}-pre.txt",
            issues_name=f"alignment-{stem}-pre-issues.json",
            checkpoints=checkpoints,
        )
        print(f"  pre-align → {pre_srt} ({pre_warnings} alignment flags)", flush=True)

        write_srt(chunks, srt_path)
        (ep_dir / f"alignment-{stem}.txt").write_text(
            (ep_dir / f"alignment-{stem}-pre.txt").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (ep_dir / f"alignment-{stem}-issues.json").write_text(
            (ep_dir / f"alignment-{stem}-pre-issues.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        txt_path.write_text(text.strip(), encoding="utf-8")

        if pre_warnings:
            print(
                f"  ALIGN WARNINGS ({pre_warnings}) → {ep_dir / f'alignment-{stem}.txt'}",
                flush=True,
            )
        else:
            print(f"  alignment OK → {ep_dir / f'alignment-{stem}.txt'}", flush=True)

        meta = {
            "source": str(src),
            "duration_sec": duration_sec,
            "chunks": len(chunks),
            "chars": len(text),
            "model": args.model,
            "model_dir": asr.model_dir,
            "engine": "sensevoice-onnx",
            "vad": not args.no_vad,
            "short_gap_vad": not args.no_short_gap_vad,
            **vad_stats,
            "alignment_warnings": pre_warnings,
        }
        (ep_dir / f"{stem}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  OK → {srt_path} ({len(chunks)} cues, {len(text)} chars)", flush=True)
        if text:
            print(f"  preview: {text[:120]}…", flush=True)

    if failed:
        print("FAILED:", ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
