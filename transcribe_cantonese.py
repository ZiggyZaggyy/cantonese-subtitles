#!/usr/bin/env python3
"""Cantonese TV → SRT via reachan/Cantonese-Whisper-Medium (Intel Mac / CPU).

Uses Silero VAD to transcribe speech segments only (skips music/gaps) and filters
Whisper repetition / song hallucinations before writing SRT.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import pipeline

DEFAULT_MODEL = "reachan/Cantonese-Whisper-Medium"
DEFAULT_PROCESSOR = "openai/whisper-medium"
# Best local Cantonese large-v3 we tested (slower than medium; pass via --model).
LARGE_V3_MODEL = "awong-dev/whisper-large-v3-yue-lora-dec-enc4"
LARGE_V3_PROCESSOR = "openai/whisper-large-v3"
SAMPLE_RATE = 16000
# Max audio length per Whisper pass (VAD regions are split before ASR). NOT on-screen
# subtitle duration — see MAX_CUE_SEC. TV dialogue rarely exceeds this; the 14s library
# region (v4/v5) was the failure case — 10s splits it into two passes. Shorter only
# adds CPU time with little gain; 5s on screen is enforced separately below.
MAX_SEGMENT_SEC = 10.0
MERGE_GAP_SEC = 0.35
MAX_CHARS_PER_CUE = 42
# Max seconds a single subtitle cue stays on screen (readability).
MAX_CUE_SEC = 5.0
# Rare cues may exceed MAX_CUE_SEC when Whisper returns one unbreakable span
# (e.g. drawn-out syllable); refine_chunks keeps them rather than forcing a split.
MAX_CUE_SEC_HARD = 7.0
PAUSE_STRONG_SEC = 0.45
PAUSE_WEAK_SEC = 0.18
MIN_CUE_SEC = 0.8
# Minimum gap only when fixing overlapping cues (not a target spacing — avoids
# cumulative +0.42s delay that made v6 subtitles run late from 3:12 onward).
MIN_OVERLAP_FIX_SEC = 0.06
PAUSE_STRONG_GAP_SEC = 0.35
# Silero VAD misses quiet TV dialogue on long files; re-scan large gaps locally.
MIN_GAP_FILL_SEC = 4.0
GAP_FILL_THRESHOLD = 0.22
GAP_FILL_PAD_MS = 400
REGION_PAD_SEC = 0.45
# Small nudge when Whisper runs slightly early; v6 used 0.30 and read late vs audio.
DEFAULT_CUE_SHIFT_SEC = 0.08
# Reject faint VAD hits (music/breath) before sending to Whisper.
MIN_REGION_RMS = 0.014
MIN_VAD_REGION_SEC = 0.85
# Faint dialogue in gaps between VAD regions (e.g. 阿哥呢, 八十五蚊).
MIN_GAP_SPEECH_RMS = 0.009
MIN_INTER_SEGMENT_GAP_SEC = 0.25
MAX_INTER_SEGMENT_GAP_SEC = 4.5
# Silero misses speech in long-context scans but finds it on isolated short gaps.
SHORT_GAP_VAD_MIN_SPEECH_SEC = 0.25
SHORT_GAP_VAD_MIN_SPEECH_MS = 150
SHORT_GAP_VAD_MIN_SILENCE_MS = 150
SHORT_GAP_VAD_PAD_MS = 200
# Fill speech missed between subtitle cues (e.g. 4:32–4:41, 4:52–4:57 laughs).
MIN_TIMELINE_GAP_SEC = 1.0
MAX_TIMELINE_GAP_SEC = 14.0
MIN_TIMELINE_GAP_RMS = 0.010
# Short gap-fill lines (阿哥呢) — Whisper timestamps run early; bias into latter half of gap.
GAP_FILL_START_BIAS = 0.38
GAP_FILL_MAX_CUE_SEC = 2.2
# Trailing bridge tail uses only the last fraction of the parent cue span.
TRAILING_BRIDGE_TAIL_FRAC = 0.20
TRAILING_BRIDGE_TAIL_MIN_SEC = 0.75
TRAILING_BRIDGE_MARKERS = ("咪理佢啫",)
CLAUSE_BRIDGE_GAP_MAX = 2.0
CLAUSE_BRIDGE_TAIL = "啫喇啦呀，"
# Stricter char density for very short regions (hallucinations pack text into <2s).
SHORT_REGION_MAX_SEC = 2.0
SHORT_REGION_MAX_CPS = 9.0
# Rough CPU multiplier vs speech duration (medium on Intel x86, VAD path).
CPU_MINUTES_PER_AUDIO_MINUTE = 4
CPU_MINUTES_PER_AUDIO_MINUTE_LARGE = 12


def cpu_minutes_per_audio_minute(model: str) -> float:
    if "large" in model.lower():
        return CPU_MINUTES_PER_AUDIO_MINUTE_LARGE
    return CPU_MINUTES_PER_AUDIO_MINUTE

WHISPER_GENERATE_KWARGS = {
    "language": "chinese",
    "task": "transcribe",
    "condition_on_prev_tokens": False,
}
# If Whisper chunk span < this fraction of VAD region, redistribute on full region.
REGION_MIN_TIMESTAMP_COVERAGE = 0.65

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
REPEAT_CHAR_RE = re.compile(r"(.)\1{9,}")
# Bilibili / stream outros Whisper hallucinates over music or silence.
STREAMING_OUTRO_RE = re.compile(
    r"(感谢|多謝|多谢|谢谢收看|訂閱|订阅).{0,10}"
    r"(关注|订阅|收看|点赞|分享|留言|频道|頻道|我们下|下次见)"
)
# Phrases Whisper invents that never appear in 1970s TV drama dialogue.
KNOWN_HALLUCINATION_PHRASES = (
    "感谢大家的关注",
    "多謝大家嘅收看",
    "多谢大家的观看",
    "请不吝点赞",
    "路上有很多车站",
    "字幕由",
    "subtitle",
)
# Strong sentence end vs weaker clause pause (keep delimiter on preceding unit).
SPLIT_STRONG_RE = re.compile(r"(?<=[。！？!?．])")
SPLIT_WEAK_RE = re.compile(r"(?<=[，、；])")
# Cantonese often has no full stops; clause-final particles mark short pauses.
SPLIT_CLAUSE_RE = re.compile(r"(?<=[啫噃喇啦])")


def extract_wav(video: Path, wav: Path, start_sec: float = 0, duration_sec: float | None = None) -> None:
    cmd = ["ffmpeg", "-nostdin", "-y"]
    if start_sec > 0:
        cmd.extend(["-ss", str(start_sec)])
    cmd.extend(["-i", str(video), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE)])
    if duration_sec is not None:
        cmd.extend(["-t", str(duration_sec)])
    cmd.append(str(wav))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_audio(path: Path) -> tuple[np.ndarray, float]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {sr}")
    duration_sec = len(audio) / SAMPLE_RATE
    return audio, duration_sec


def load_vad():
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    return model, utils[0]


def normalize_segments(
    segments: list[tuple[float, float]],
    *,
    max_segment_sec: float = MAX_SEGMENT_SEC,
    merge_gap_sec: float = MERGE_GAP_SEC,
) -> list[tuple[float, float]]:
    """Merge nearby intervals and split segments longer than max_segment_sec."""
    if not segments:
        return []

    merged: list[tuple[int, int]] = []
    for start_f, end_f in sorted(segments):
        start, end = int(start_f * SAMPLE_RATE), int(end_f * SAMPLE_RATE)
        if end <= start:
            continue
        if merged and start - merged[-1][1] <= int(merge_gap_sec * SAMPLE_RATE):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    max_samples = int(max_segment_sec * SAMPLE_RATE)
    out: list[tuple[float, float]] = []
    for start, end in merged:
        while end - start > max_samples:
            out.append((start / SAMPLE_RATE, (start + max_samples) / SAMPLE_RATE))
            start += max_samples
        if end > start:
            out.append((start / SAMPLE_RATE, end / SAMPLE_RATE))
    return out


def fill_vad_gaps(
    audio: np.ndarray,
    segments: list[tuple[float, float]],
    vad_model,
    get_speech_timestamps,
    *,
    min_gap_sec: float = MIN_GAP_FILL_SEC,
    gap_threshold: float = GAP_FILL_THRESHOLD,
    speech_pad_ms: int = GAP_FILL_PAD_MS,
) -> list[tuple[float, float]]:
    """Re-run VAD on long gaps — catches quiet dialogue missed on full-file pass."""
    if not segments:
        return segments

    sorted_segs = sorted(segments)
    gaps: list[tuple[float, float]] = []
    if sorted_segs[0][0] >= min_gap_sec:
        gaps.append((0.0, sorted_segs[0][0]))
    for i in range(len(sorted_segs) - 1):
        gap_start, gap_end = sorted_segs[i][1], sorted_segs[i + 1][0]
        if gap_end - gap_start >= min_gap_sec:
            gaps.append((gap_start, gap_end))

    extra: list[tuple[float, float]] = []
    for gap_start, gap_end in gaps:
        s0, s1 = int(gap_start * SAMPLE_RATE), int(gap_end * SAMPLE_RATE)
        gap_audio = audio[s0:s1]
        if len(gap_audio) < int(0.4 * SAMPLE_RATE):
            continue
        stamps = get_speech_timestamps(
            torch.from_numpy(gap_audio),
            vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=gap_threshold,
            min_speech_duration_ms=150,
            min_silence_duration_ms=150,
            speech_pad_ms=speech_pad_ms,
        )
        for ts in stamps or []:
            abs_start = gap_start + ts["start"] / SAMPLE_RATE
            abs_end = gap_start + ts["end"] / SAMPLE_RATE
            if abs_end > abs_start:
                extra.append((abs_start, abs_end))

    if not extra:
        return segments
    return sorted_segs + extra


def _gap_inside_recovered(
    gap_start: float,
    gap_end: float,
    recovered: list[tuple[float, float]],
) -> bool:
    """True if this inter-segment gap lies inside a short-gap VAD parent."""
    for parent_start, parent_end in recovered:
        if gap_start >= parent_start - 0.05 and gap_end <= parent_end + 0.05:
            return True
    return False


def expand_short_gap_vad(
    audio: np.ndarray,
    segments: list[tuple[float, float]],
    vad_model,
    get_speech_timestamps,
    *,
    gap_threshold: float = GAP_FILL_THRESHOLD,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Re-run VAD on short inter-segment gaps (isolated context).

    Silero often misses loud dialogue stranded between two blobs when scanned
    in a long parent gap, but detects it on the 1–4s slice alone (e.g. 阿哥呢).
    Returns expanded segments and parent-gap bounds for skipping duplicate gap-fill.
    """
    if len(segments) < 2:
        return segments, []

    sorted_segs = sorted(segments)
    extra: list[tuple[float, float]] = []
    recovered_parents: list[tuple[float, float]] = []

    for i in range(len(sorted_segs) - 1):
        gap_start = sorted_segs[i][1]
        gap_end = sorted_segs[i + 1][0]
        gap_dur = gap_end - gap_start
        if gap_dur < MIN_INTER_SEGMENT_GAP_SEC or gap_dur > MAX_INTER_SEGMENT_GAP_SEC:
            continue
        if region_rms(audio, gap_start, gap_end) < MIN_GAP_SPEECH_RMS:
            continue

        s0 = int(gap_start * SAMPLE_RATE)
        s1 = int(gap_end * SAMPLE_RATE)
        gap_audio = audio[s0:s1]
        if len(gap_audio) < int(MIN_INTER_SEGMENT_GAP_SEC * SAMPLE_RATE):
            continue

        stamps = get_speech_timestamps(
            torch.from_numpy(gap_audio),
            vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=gap_threshold,
            min_speech_duration_ms=SHORT_GAP_VAD_MIN_SPEECH_MS,
            min_silence_duration_ms=SHORT_GAP_VAD_MIN_SILENCE_MS,
            speech_pad_ms=SHORT_GAP_VAD_PAD_MS,
        )
        if not stamps:
            continue

        rel_starts = [ts["start"] / SAMPLE_RATE for ts in stamps]
        rel_ends = [ts["end"] / SAMPLE_RATE for ts in stamps]
        vad_start = gap_start + min(rel_starts)
        vad_end = gap_start + max(rel_ends)
        speech_dur = vad_end - vad_start
        if speech_dur < SHORT_GAP_VAD_MIN_SPEECH_SEC:
            continue

        if speech_dur < MIN_VAD_REGION_SEC:
            pad = (MIN_VAD_REGION_SEC - speech_dur) / 2
            vad_start = max(gap_start, vad_start - pad)
            vad_end = min(gap_end, vad_end + pad)

        extra.append((vad_start, vad_end))
        recovered_parents.append((gap_start, gap_end))
        print(
            f"  short-gap VAD [{gap_start:.1f}s–{gap_end:.1f}s] rms="
            f"{region_rms(audio, gap_start, gap_end):.4f} → "
            f"region [{vad_start:.1f}s–{vad_end:.1f}s]",
            flush=True,
        )

    if not extra:
        return segments, []
    return sorted_segs + extra, recovered_parents


def speech_segments(
    audio: np.ndarray,
    vad_model,
    get_speech_timestamps,
    *,
    max_segment_sec: float = MAX_SEGMENT_SEC,
    merge_gap_sec: float = MERGE_GAP_SEC,
    vad_threshold: float = 0.35,
    speech_pad_ms: int = 250,
    gap_fill: bool = True,
    min_gap_fill_sec: float = MIN_GAP_FILL_SEC,
    gap_fill_threshold: float = GAP_FILL_THRESHOLD,
    short_gap_vad: bool = True,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Return speech regions and short-gap VAD parent bounds (for gap-fill dedup)."""
    wav = torch.from_numpy(audio)
    stamps = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=vad_threshold,
        min_speech_duration_ms=200,
        min_silence_duration_ms=200,
        speech_pad_ms=speech_pad_ms,
    )
    if not stamps:
        return [], []

    initial: list[tuple[float, float]] = []
    for ts in stamps:
        start = ts["start"] / SAMPLE_RATE
        end = ts["end"] / SAMPLE_RATE
        if end > start:
            initial.append((start, end))

    if gap_fill:
        initial = fill_vad_gaps(
            audio,
            initial,
            vad_model,
            get_speech_timestamps,
            min_gap_sec=min_gap_fill_sec,
            gap_threshold=gap_fill_threshold,
            speech_pad_ms=GAP_FILL_PAD_MS,
        )

    normalized = normalize_segments(
        initial,
        max_segment_sec=max_segment_sec,
        merge_gap_sec=merge_gap_sec,
    )

    recovered_parents: list[tuple[float, float]] = []
    if short_gap_vad and len(normalized) >= 2:
        expanded, recovered_parents = expand_short_gap_vad(
            audio,
            normalized,
            vad_model,
            get_speech_timestamps,
            gap_threshold=gap_fill_threshold,
        )
        normalized = normalize_segments(
            expanded,
            max_segment_sec=max_segment_sec,
            merge_gap_sec=merge_gap_sec,
        )

    return normalized, recovered_parents


def cjk_ratio(text: str) -> float:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return 0.0
    return len(CJK_RE.findall(compact)) / len(compact)


def has_phrase_loop(text: str, min_repeats: int = 4) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 12:
        return False
    for size in range(2, 9):
        for i in range(0, len(compact) - size * min_repeats + 1):
            phrase = compact[i : i + size]
            if len(phrase.strip()) < 2:
                continue
            reps = 1
            pos = i + size
            while compact.startswith(phrase, pos):
                reps += 1
                pos += size
                if reps >= min_repeats:
                    return True
    return False


def region_rms(audio: np.ndarray, start_sec: float, end_sec: float) -> float:
    s0 = max(0, int(start_sec * SAMPLE_RATE))
    s1 = min(len(audio), int(end_sec * SAMPLE_RATE))
    chunk = audio[s0:s1]
    if len(chunk) == 0:
        return 0.0
    return float(np.sqrt(np.mean(chunk * chunk)))


def is_hallucination(
    text: str,
    duration_sec: float,
    *,
    region_dur: float | None = None,
) -> bool:
    text = text.strip()
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 2:
        return True
    if REPEAT_CHAR_RE.search(text):
        return True
    if has_phrase_loop(text):
        return True
    for phrase in KNOWN_HALLUCINATION_PHRASES:
        if phrase in compact:
            return True
    cps = len(compact) / max(duration_sec, 0.3)
    if cps > 18:
        return True
    ref_dur = region_dur if region_dur is not None else duration_sec
    if ref_dur < SHORT_REGION_MAX_SEC and len(compact) / max(ref_dur, 0.3) > SHORT_REGION_MAX_CPS:
        return True
    if len(text) > 40 and cjk_ratio(text) < 0.25:
        return True
    if text.count("chambre") > 2 or text.count("Tu vas") > 2:
        return True
    if STREAMING_OUTRO_RE.search(compact):
        return True
    return False


def should_skip_vad_region(
    audio: np.ndarray,
    start_sec: float,
    end_sec: float,
) -> str | None:
    """Return rejection reason, or None if region should be transcribed."""
    dur = end_sec - start_sec
    if dur < MIN_VAD_REGION_SEC:
        return f"too short ({dur:.2f}s)"
    rms = region_rms(audio, start_sec, end_sec)
    if rms < MIN_REGION_RMS:
        return f"low energy (rms={rms:.4f})"
    return None


def _split_long_unit(text: str, max_chars: int) -> list[str]:
    """Hard-wrap a single unit that still exceeds max_chars."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    parts: list[str] = []
    while len(text) > max_chars:
        cut = max_chars
        window = text[: max_chars + 1]
        weak = max(window.rfind("，"), window.rfind("、"), window.rfind("；"), window.rfind(" "))
        if weak >= max_chars // 3:
            cut = weak + 1
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts


def text_units(text: str, *, max_chars: int = MAX_CHARS_PER_CUE) -> list[tuple[str, str]]:
    """Split transcript into subtitle units with pause strength ('strong'|'weak'|'none')."""
    text = re.sub(r"\s+", "", text.strip())
    if not text:
        return []

    def pause_for(unit: str) -> str:
        if not unit:
            return "none"
        if unit[-1] in "。！？!?．":
            return "strong"
        if unit[-1] in "，、；啫噃喇啦":
            return "weak"
        return "none"

    def split_pass(parts: list[str], pattern: re.Pattern) -> list[str]:
        out: list[str] = []
        for part in parts:
            bits = [b for b in pattern.split(part) if b]
            out.extend(bits if bits else [part])
        return out

    parts = [text]
    for pattern in (SPLIT_STRONG_RE, SPLIT_WEAK_RE, SPLIT_CLAUSE_RE):
        next_parts = split_pass(parts, pattern)
        if len(next_parts) > len(parts):
            parts = next_parts

    units: list[tuple[str, str]] = []
    for part in parts:
        for piece in _split_long_unit(part, max_chars):
            if piece:
                units.append((piece, pause_for(piece)))
    return units


def group_units(
    units: list[tuple[str, str]],
    *,
    max_chars: int = MAX_CHARS_PER_CUE,
) -> list[tuple[str, list[str]]]:
    """Merge adjacent units into cues, breaking at pauses when over char limit."""
    if not units:
        return []

    groups: list[tuple[str, list[str]]] = []
    cur_text = ""
    cur_pauses: list[str] = []

    def flush() -> None:
        nonlocal cur_text, cur_pauses
        if cur_text:
            groups.append((cur_text, cur_pauses))
        cur_text = ""
        cur_pauses = []

    for unit, pause in units:
        if cur_text and len(cur_text) + len(unit) > max_chars:
            flush()
        if not cur_text and len(unit) > max_chars:
            groups.append((unit, []))
            continue
        cur_text += unit
        if pause != "none":
            cur_pauses.append(pause)
    flush()
    return groups


def allocate_timestamps(
    groups: list[tuple[str, list[str]]],
    start: float,
    end: float,
    *,
    max_cue_sec: float = MAX_CUE_SEC,
    pause_strong_sec: float = PAUSE_STRONG_SEC,
    pause_weak_sec: float = PAUSE_WEAK_SEC,
    min_cue_sec: float = MIN_CUE_SEC,
) -> list[tuple[float, float, str]]:
    """Distribute cue times by text length + weighted pauses between grouped units."""
    if not groups:
        return []
    total_dur = max(end - start, min_cue_sec)

    weights: list[float] = []
    for text, pauses in groups:
        w = max(len(text), 1)
        for p in pauses:
            w += (pause_strong_sec if p == "strong" else pause_weak_sec) * 12.0
        weights.append(w)

    total_w = sum(weights) or 1.0
    out: list[tuple[float, float, str]] = []
    pos = start
    remaining = total_dur
    remaining_w = total_w
    for i, ((text, _), w) in enumerate(zip(groups, weights)):
        if i == len(groups) - 1:
            cue_dur = remaining
        else:
            share = (w / remaining_w) * remaining if remaining_w > 0 else remaining / (len(groups) - i)
            cue_dur = min(max(share, min_cue_sec), max_cue_sec, remaining)
        cue_end = min(pos + max(cue_dur, min_cue_sec), end)
        if cue_end <= pos:
            cue_end = min(pos + min_cue_sec, end)
        out.append((pos, cue_end, text))
        used = cue_end - pos
        remaining = max(0.0, remaining - used)
        remaining_w = max(0.0, remaining_w - w)
        pos = cue_end
        if i < len(groups) - 1:
            pauses = groups[i][1]
            if pauses and pauses[-1] == "strong":
                pos += PAUSE_STRONG_GAP_SEC
            elif pauses:
                pos += PAUSE_WEAK_SEC * 0.4
    return out


def scale_chunks_to_region(
    chunks: list[dict],
    region_start: float,
    region_end: float,
    *,
    cue_shift_sec: float = 0.0,
) -> list[dict]:
    """Linearly scale Whisper chunk timestamps to cover the full VAD region."""
    if not chunks:
        return chunks
    r_start = region_start + cue_shift_sec
    r_end = region_end + cue_shift_sec
    region_dur = r_end - r_start
    c_start = chunks[0]["timestamp"][0]
    c_end = chunks[-1]["timestamp"][1]
    span = c_end - c_start
    if span < 0.05 or span >= region_dur * REGION_MIN_TIMESTAMP_COVERAGE:
        return chunks
    scale = region_dur / span
    out: list[dict] = []
    for c in chunks:
        rel0 = c["timestamp"][0] - c_start
        rel1 = c["timestamp"][1] - c_start
        out.append(
            {
                "timestamp": (r_start + rel0 * scale, r_start + rel1 * scale),
                "text": c["text"],
            }
        )
    return out


def enforce_cue_separation(
    chunks: list[dict],
    *,
    overlap_fix_sec: float = MIN_OVERLAP_FIX_SEC,
) -> list[dict]:
    """Fix overlapping cues only — do not push starts later when already separated."""
    if not chunks:
        return chunks
    sorted_chunks = sorted(chunks, key=lambda c: c["timestamp"][0])
    out: list[dict] = []
    for c in sorted_chunks:
        start, end = c["timestamp"]
        text = c["text"]
        if out:
            prev_end = out[-1]["timestamp"][1]
            if start < prev_end - 0.02:
                start = prev_end + overlap_fix_sec
        if end <= start:
            end = start + MIN_CUE_SEC
        out.append({"timestamp": (start, end), "text": text})
    return out


def split_trailing_bridge_cues(chunks: list[dict]) -> list[dict]:
    """Pull trailing bridge phrases into a late sub-cue (e.g. …咪理佢啫 before next line)."""
    out: list[dict] = []
    for c in chunks:
        text = c["text"].strip()
        start, end = c["timestamp"]
        marker = next(
            (m for m in TRAILING_BRIDGE_MARKERS if text.endswith(m) and len(text) > len(m) + 6),
            None,
        )
        if not marker:
            out.append(c)
            continue
        head = text[: -len(marker)].rstrip("，")
        span = end - start
        tail_dur = max(
            TRAILING_BRIDGE_TAIL_MIN_SEC,
            min(1.1, span * TRAILING_BRIDGE_TAIL_FRAC),
        )
        head_end = end - tail_dur - 0.05
        if head_end <= start + MIN_CUE_SEC or not head:
            out.append(c)
            continue
        out.append({"timestamp": (start, head_end), "text": head})
        out.append({"timestamp": (head_end + 0.04, end), "text": marker})
    return out


def bridge_adjacent_cues(chunks: list[dict]) -> list[dict]:
    """Extend clause-final cues to the next line (e.g. 咪理佢啫 → 大幫我賣件喇)."""
    if not chunks:
        return chunks
    out: list[dict] = []
    for c in sorted(chunks, key=lambda x: x["timestamp"][0]):
        out.append({"timestamp": tuple(c["timestamp"]), "text": c["text"]})
    for i in range(len(out) - 1):
        start_i, end_i = out[i]["timestamp"]
        start_next, _ = out[i + 1]["timestamp"]
        text = out[i]["text"].strip()
        gap = start_next - end_i
        if gap < -0.05 or gap > CLAUSE_BRIDGE_GAP_MAX:
            continue
        if text and (text[-1] in CLAUSE_BRIDGE_TAIL or text.endswith("咪理佢啫")):
            new_end = max(end_i, start_next - 0.05)
            if new_end > start_i + MIN_CUE_SEC:
                out[i]["timestamp"] = (start_i, new_end)
    return out


def align_chunks_to_region(
    chunks: list[dict],
    region_start: float,
    region_end: float,
    *,
    cue_shift_sec: float = 0.0,
) -> list[dict]:
    """Spread cues across a VAD region when Whisper timestamps under-cover it."""
    if not chunks:
        return chunks
    r_start = region_start + cue_shift_sec
    r_end = region_end + cue_shift_sec
    region_dur = r_end - r_start
    chunk_span = chunks[-1]["timestamp"][1] - chunks[0]["timestamp"][0]
    if chunk_span >= region_dur * REGION_MIN_TIMESTAMP_COVERAGE:
        return chunks
    if len(chunks) > 1:
        return scale_chunks_to_region(
            chunks, region_start, region_end, cue_shift_sec=cue_shift_sec
        )
    merged = chunks[0]["text"].strip()
    if not merged:
        return chunks
    units = text_units(merged)
    groups = group_units(units)
    if not groups:
        return chunks
    out: list[dict] = []
    for sub_start, sub_end, sub_text in allocate_timestamps(groups, r_start, r_end):
        if sub_text and sub_end > sub_start:
            out.append({"timestamp": (sub_start, sub_end), "text": sub_text})
    return out if out else chunks


def refine_chunks(
    chunks: list[dict],
    *,
    max_chars: int = MAX_CHARS_PER_CUE,
    max_cue_sec: float = MAX_CUE_SEC,
) -> list[dict]:
    """Split long cues on sentence/clause pauses; cap per-cue duration."""
    refined: list[dict] = []
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        ts = chunk.get("timestamp")
        if not text or not ts:
            continue
        start = ts[0]
        end = ts[1] if len(ts) > 1 and ts[1] is not None else start + 2.0
        if end <= start:
            end = start + MIN_CUE_SEC

        span = end - start
        min_cues = max(1, math.ceil(span / max_cue_sec))
        chars = max_chars
        groups: list[tuple[str, list[str]]] = []
        while chars >= 12:
            units = text_units(text, max_chars=chars)
            groups = group_units(units, max_chars=chars)
            if len(groups) >= min_cues or chars <= 12:
                break
            chars = max(12, int(chars * 0.75))
        if not groups:
            continue

        if len(groups) == 1 and span <= MAX_CUE_SEC_HARD:
            refined.append({"timestamp": (start, end), "text": groups[0][0]})
            continue

        for sub_start, sub_end, sub_text in allocate_timestamps(
            groups, start, end, max_cue_sec=max_cue_sec
        ):
            if sub_text and sub_end > sub_start:
                refined.append({"timestamp": (sub_start, sub_end), "text": sub_text})

    refined.sort(key=lambda c: c["timestamp"][0])
    return refined


def srt_time(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - math.floor(sec)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(chunks: list[dict], path: Path, *, max_cue_sec: float = MAX_CUE_SEC) -> None:
    lines: list[str] = []
    n = 1
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        ts = chunk.get("timestamp")
        if not text or not ts:
            continue
        start = ts[0] if len(ts) > 0 else None
        end = ts[1] if len(ts) > 1 else None
        if start is None:
            continue
        if end is None:
            end = start + min(max(len(text) / 10.0, 1.5), max_cue_sec)
        if end <= start:
            end = start + min(max(len(text) / 10.0, 1.5), max_cue_sec)
        lines.append(str(n))
        lines.append(f"{srt_time(start)} --> {srt_time(end)}")
        lines.append(text)
        lines.append("")
        n += 1
    path.write_text("\n".join(lines), encoding="utf-8")


def episode_num(name: str) -> int:
    for pat, n in [
        (r"第一集", 1),
        (r"第二集", 2),
        (r"第三集", 3),
        (r"第四集", 4),
        (r"第五集", 5),
        (r"第六集", 6),
        (r"完结篇", 7),
    ]:
        if re.search(pat, name):
            return n
    return 0


def verify_result(text: str, duration_sec: float, speech_sec: float | None = None) -> bool:
    ref = speech_sec if speech_sec and speech_sec > 0 else duration_sec
    if len(text.strip()) < max(40, ref * 0.35):
        return False
    if is_hallucination(text, max(ref, 5.0)):
        return False
    return True


def build_speech_track(
    audio: np.ndarray,
    segments: list[tuple[float, float]],
) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
    """Concatenate speech; return audio + (concat_start, concat_end, abs_start, abs_end)."""
    pieces: list[np.ndarray] = []
    mapping: list[tuple[float, float, float, float]] = []
    pos = 0.0
    for abs_start, abs_end in segments:
        s0, s1 = int(abs_start * SAMPLE_RATE), int(abs_end * SAMPLE_RATE)
        piece = audio[s0:s1]
        if len(piece) == 0:
            continue
        dur = len(piece) / SAMPLE_RATE
        pieces.append(piece)
        mapping.append((pos, pos + dur, abs_start, abs_end))
        pos += dur
    if not pieces:
        return np.array([], dtype=np.float32), []
    return np.concatenate(pieces), mapping


def map_to_absolute(time_sec: float, mapping: list[tuple[float, float, float, float]]) -> float:
    for c_start, c_end, a_start, a_end in mapping:
        if c_start <= time_sec <= c_end:
            return a_start + (time_sec - c_start)
        if time_sec < c_start:
            return a_start
    if mapping:
        return mapping[-1][3]
    return time_sec


def chunks_from_result(
    result: dict,
    mapping: list[tuple[float, float, float, float]],
    *,
    concat_duration: float,
) -> list[dict]:
    out: list[dict] = []
    for chunk in result.get("chunks") or []:
        text = (chunk.get("text") or "").strip()
        ts = chunk.get("timestamp")
        if not text or not ts:
            continue
        rel_start = ts[0] if len(ts) > 0 else None
        rel_end = ts[1] if len(ts) > 1 else None
        if rel_start is None:
            continue
        if rel_end is None:
            rel_end = rel_start + min(max(len(text) / 12.0, 1.0), MAX_CUE_SEC)
        abs_start = map_to_absolute(rel_start, mapping)
        abs_end = map_to_absolute(min(rel_end, concat_duration), mapping)
        if abs_end <= abs_start:
            abs_end = abs_start + 1.0
        if is_hallucination(text, abs_end - abs_start):
            continue
        out.append({"timestamp": (abs_start, abs_end), "text": text})

    if not out:
        text = (result.get("text") or "").strip()
        if text and mapping and not is_hallucination(text, mapping[-1][3] - mapping[0][2]):
            out.append({"timestamp": (mapping[0][2], mapping[-1][3]), "text": text})
    return out


def merge_chunks_into_regions(
    chunks: list[dict],
    segments: list[tuple[float, float]],
) -> list[dict]:
    """Collapse whisper chunks into one cue per VAD speech region."""
    if not segments:
        return chunks
    merged: list[dict] = []
    for seg_start, seg_end in segments:
        in_seg = [
            c
            for c in chunks
            if c["timestamp"][0] >= seg_start - 0.2
            and c["timestamp"][1] <= seg_end + 0.5
        ]
        if not in_seg:
            continue
        text = "".join(c["text"] for c in in_seg).strip()
        dur = seg_end - seg_start
        if not text or is_hallucination(text, dur):
            continue
        merged.append({"timestamp": (seg_start, seg_end), "text": text})
    return merged if merged else chunks


def transcribe_region(
    asr,
    audio: np.ndarray,
    offset_sec: float,
    *,
    region_start: float | None = None,
    region_end: float | None = None,
    cue_shift_sec: float = 0.0,
) -> list[dict]:
    """Transcribe one VAD region; timestamps are absolute in the source video."""
    seg_dur = len(audio) / SAMPLE_RATE
    if seg_dur < 0.2:
        return []
    r_start = region_start if region_start is not None else offset_sec
    r_end = region_end if region_end is not None else offset_sec + seg_dur

    result = asr(
        {"array": audio, "sampling_rate": SAMPLE_RATE},
        return_timestamps=True,
        generate_kwargs=dict(WHISPER_GENERATE_KWARGS),
    )

    chunks: list[dict] = []
    for chunk in result.get("chunks") or []:
        text = (chunk.get("text") or "").strip()
        ts = chunk.get("timestamp")
        if not text or not ts:
            continue
        rel_start = ts[0] if len(ts) > 0 else None
        rel_end = ts[1] if len(ts) > 1 else None
        if rel_start is None:
            continue
        if rel_end is None:
            rel_end = rel_start + min(max(len(text) / 10.0, 1.0), seg_dur)
        rel_end = min(rel_end, seg_dur)
        abs_start = max(r_start, offset_sec + rel_start) + cue_shift_sec
        abs_end = min(r_end, offset_sec + rel_end) + cue_shift_sec
        if abs_end <= abs_start:
            abs_end = abs_start + MIN_CUE_SEC
        region_dur = r_end - r_start
        if is_hallucination(text, abs_end - abs_start, region_dur=region_dur):
            continue
        chunks.append({"timestamp": (abs_start, abs_end), "text": text})

    if not chunks:
        text = (result.get("text") or "").strip()
        region_dur = r_end - r_start
        if text and not is_hallucination(text, seg_dur, region_dur=region_dur):
            chunks.append(
                {
                    "timestamp": (r_start + cue_shift_sec, r_end + cue_shift_sec),
                    "text": text,
                }
            )
    return chunks


def bias_gap_fill_timestamps(
    chunks: list[dict],
    gap_start: float,
    gap_end: float,
) -> list[dict]:
    """Place short gap-fill lines later in the gap (Whisper onset runs early)."""
    gap_dur = gap_end - gap_start
    if gap_dur < 0.3:
        return chunks
    out: list[dict] = []
    for c in chunks:
        text = (c.get("text") or "").strip()
        if len(text) > 14:
            out.append(c)
            continue
        cue_dur = min(GAP_FILL_MAX_CUE_SEC, max(MIN_CUE_SEC, gap_dur * 0.55))
        t0 = gap_start + gap_dur * GAP_FILL_START_BIAS
        t1 = min(gap_end, t0 + cue_dur)
        if t1 <= t0:
            t1 = t0 + MIN_CUE_SEC
        out.append({"timestamp": (t0, t1), "text": text})
    return out


def transcribe_inter_segment_gaps(
    asr,
    audio: np.ndarray,
    segments: list[tuple[float, float]],
    *,
    cue_shift_sec: float = 0.0,
    skip_recovered_gaps: list[tuple[float, float]] | None = None,
) -> list[dict]:
    """Transcribe short speech in gaps between VAD regions (missed faint dialogue)."""
    if len(segments) < 2:
        return []
    recovered = skip_recovered_gaps or []
    extra: list[dict] = []
    for i in range(len(segments) - 1):
        gap_start = segments[i][1]
        gap_end = segments[i + 1][0]
        gap_dur = gap_end - gap_start
        if gap_dur < MIN_INTER_SEGMENT_GAP_SEC or gap_dur > MAX_INTER_SEGMENT_GAP_SEC:
            continue
        if _gap_inside_recovered(gap_start, gap_end, recovered):
            continue
        rms = region_rms(audio, gap_start, gap_end)
        if rms < MIN_GAP_SPEECH_RMS:
            continue
        s0 = int(gap_start * SAMPLE_RATE)
        s1 = int(gap_end * SAMPLE_RATE)
        seg_audio = audio[s0:s1]
        if len(seg_audio) < int(0.2 * SAMPLE_RATE):
            continue
        chunks = transcribe_region(
            asr,
            seg_audio,
            gap_start,
            region_start=gap_start,
            region_end=gap_end,
            cue_shift_sec=cue_shift_sec,
        )
        chunks = bias_gap_fill_timestamps(chunks, gap_start, gap_end)
        if chunks:
            print(
                f"  gap fill [{gap_start:.1f}s–{gap_end:.1f}s] rms={rms:.4f}: "
                f"{chunks[0]['text'][:40]}",
                flush=True,
            )
            extra.extend(chunks)
    return extra


def transcribe_timeline_gaps(
    asr,
    audio: np.ndarray,
    chunks: list[dict],
    *,
    cue_shift_sec: float = 0.0,
) -> list[dict]:
    """Transcribe speech in long gaps between subtitle cues (within same scene)."""
    if len(chunks) < 2:
        return []
    sorted_c = sorted(chunks, key=lambda c: c["timestamp"][0])
    extra: list[dict] = []
    for i in range(len(sorted_c) - 1):
        gap_start = sorted_c[i]["timestamp"][1]
        gap_end = sorted_c[i + 1]["timestamp"][0]
        gap_dur = gap_end - gap_start
        if gap_dur < MIN_TIMELINE_GAP_SEC or gap_dur > MAX_TIMELINE_GAP_SEC:
            continue
        rms = region_rms(audio, gap_start, gap_end)
        if rms < MIN_TIMELINE_GAP_RMS:
            continue
        s0 = int(gap_start * SAMPLE_RATE)
        s1 = int(gap_end * SAMPLE_RATE)
        seg_audio = audio[s0:s1]
        if len(seg_audio) < int(0.25 * SAMPLE_RATE):
            continue
        new_chunks = transcribe_region(
            asr,
            seg_audio,
            gap_start,
            region_start=gap_start,
            region_end=gap_end,
            cue_shift_sec=cue_shift_sec,
        )
        for nc in new_chunks:
            nc_start = nc["timestamp"][0]
            if any(
                c["timestamp"][0] - 0.3 <= nc_start <= c["timestamp"][1] + 0.3
                for c in chunks
            ):
                continue
            print(
                f"  timeline gap [{gap_start:.1f}s–{gap_end:.1f}s] rms={rms:.4f}: "
                f"{nc['text'][:40]}",
                flush=True,
            )
            extra.append(nc)
    return extra


def transcribe_with_vad(
    asr,
    audio: np.ndarray,
    vad_model,
    get_speech_timestamps,
    *,
    vad_threshold: float = 0.35,
    speech_pad_ms: int = 250,
    gap_fill: bool = True,
    min_gap_fill_sec: float = MIN_GAP_FILL_SEC,
    gap_fill_threshold: float = GAP_FILL_THRESHOLD,
    short_gap_vad: bool = True,
    cue_shift_sec: float = DEFAULT_CUE_SHIFT_SEC,
) -> tuple[str, list[dict], dict]:
    segments, recovered_gaps = speech_segments(
        audio,
        vad_model,
        get_speech_timestamps,
        vad_threshold=vad_threshold,
        speech_pad_ms=speech_pad_ms,
        gap_fill=gap_fill,
        min_gap_fill_sec=min_gap_fill_sec,
        gap_fill_threshold=gap_fill_threshold,
        short_gap_vad=short_gap_vad,
    )
    speech_sec = sum(end - start for start, end in segments)
    print(
        f"  VAD: {len(segments)} regions, {speech_sec / 60:.1f} min speech "
        f"of {len(audio) / SAMPLE_RATE / 60:.1f} min total",
        flush=True,
    )
    if not segments:
        return "", [], {"vad_segments": 0, "speech_sec": 0.0, "rejected_segments": 0, "cues": 0}

    print(f"  transcribing each speech region separately (accurate timestamps)…", flush=True)
    duration_sec = len(audio) / SAMPLE_RATE
    all_chunks: list[dict] = []
    rejected = 0
    skipped_regions = 0
    for i, (start_sec, end_sec) in enumerate(segments, 1):
        skip_reason = should_skip_vad_region(audio, start_sec, end_sec)
        if skip_reason:
            skipped_regions += 1
            if skipped_regions <= 5:
                print(f"  skip region {i} [{start_sec:.1f}s–{end_sec:.1f}s]: {skip_reason}", flush=True)
            continue
        prev_end = segments[i - 2][1] if i > 1 else 0.0
        next_start = segments[i][0] if i < len(segments) else duration_sec
        # Pad after only — leading pad made Whisper timestamps run early vs lip-sync.
        pad_before = 0.0
        pad_after = min(REGION_PAD_SEC, max(0.0, next_start - end_sec) / 2)
        slice_start = max(0.0, start_sec - pad_before)
        slice_end = min(duration_sec, end_sec + pad_after)
        s0 = int(slice_start * SAMPLE_RATE)
        s1 = int(slice_end * SAMPLE_RATE)
        seg_audio = audio[s0:s1]
        if i == 1 or i % 15 == 0 or i == len(segments):
            print(
                f"  region {i}/{len(segments)} [{start_sec:.1f}s–{end_sec:.1f}s]"
                f" pad→[{slice_start:.1f}s–{slice_end:.1f}s]",
                flush=True,
            )
        before = len(all_chunks)
        region_chunks = transcribe_region(
            asr,
            seg_audio,
            slice_start,
            region_start=start_sec,
            region_end=end_sec,
            cue_shift_sec=cue_shift_sec,
        )
        region_chunks = align_chunks_to_region(
            region_chunks,
            start_sec,
            end_sec,
            cue_shift_sec=cue_shift_sec,
        )
        all_chunks.extend(region_chunks)
        if len(all_chunks) == before:
            rejected += 1

    gap_chunks = transcribe_inter_segment_gaps(
        asr,
        audio,
        segments,
        cue_shift_sec=cue_shift_sec,
        skip_recovered_gaps=recovered_gaps,
    )
    all_chunks.extend(gap_chunks)

    all_chunks.sort(key=lambda c: c["timestamp"][0])
    all_chunks = refine_chunks(all_chunks)
    all_chunks = split_trailing_bridge_cues(all_chunks)
    all_chunks = bridge_adjacent_cues(all_chunks)
    all_chunks = enforce_cue_separation(all_chunks)

    timeline_chunks = transcribe_timeline_gaps(
        asr, audio, all_chunks, cue_shift_sec=cue_shift_sec
    )
    all_chunks.extend(timeline_chunks)
    all_chunks.sort(key=lambda c: c["timestamp"][0])
    all_chunks = enforce_cue_separation(all_chunks)
    text = "".join(c["text"] for c in all_chunks)
    stats = {
        "vad_segments": len(segments),
        "speech_sec": round(speech_sec, 2),
        "skipped_regions": skipped_regions,
        "rejected_segments": rejected,
        "gap_cues": len(gap_chunks),
        "short_gap_vad": len(recovered_gaps),
        "timeline_gap_cues": len(timeline_chunks),
        "cues": len(all_chunks),
    }
    return text, all_chunks, stats


def finalize_alignment(
    audio: np.ndarray,
    chunks: list[dict],
    duration_sec: float,
) -> tuple[list[dict], list[str]]:
    """Optional single-pass timing polish (no gap re-whisper, no end-extension)."""
    from inspect_alignment import polish_chunks_alignment

    return polish_chunks_alignment(audio, chunks, duration_sec=duration_sec, max_passes=1)


def _write_alignment_reports(
    audio: np.ndarray,
    srt_path: Path,
    ep_dir: Path,
    *,
    duration_sec: float,
    report_name: str,
    issues_name: str,
    checkpoints: list[tuple[str, float, float]],
) -> tuple[str, int]:
    from compare_srt import parse_srt
    from inspect_alignment import (
        alignment_issues_json,
        analyze_all_cues,
        inspect_alignment,
    )

    cues = parse_srt(srt_path)
    report, warnings = inspect_alignment(
        audio, cues, duration_sec=duration_sec, checkpoints=checkpoints
    )
    report_path = ep_dir / report_name
    report_path.write_text(report, encoding="utf-8")
    analyses = analyze_all_cues(audio, cues, duration_sec=duration_sec)
    (ep_dir / issues_name).write_text(
        json.dumps(alignment_issues_json(analyses), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(report_path), len(warnings)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cantonese Whisper → SRT")
    parser.add_argument("inputs", nargs="+", help="Video or WAV files")
    parser.add_argument(
        "--out-dir",
        default=str(Path.home() / "Downloads/bilibili-videos/subtitles-yue-whisper"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--processor", default=DEFAULT_PROCESSOR)
    parser.add_argument("--start-seconds", type=float, default=0)
    parser.add_argument("--test-seconds", type=float, default=0, help="Only transcribe first N seconds")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-vad", action="store_true", help="Transcribe full audio (old behaviour)")
    parser.add_argument("--no-episode-subdir", action="store_true", help="Write out.srt directly in --out-dir")
    parser.add_argument("--vad-threshold", type=float, default=0.35)
    parser.add_argument("--speech-pad-ms", type=int, default=250)
    parser.add_argument("--no-gap-fill", action="store_true", help="Skip gap re-scan (Silero misses quiet speech)")
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
        help="Delay all cues by N seconds (fixes early subtitles vs lip-sync)",
    )
    parser.add_argument(
        "--align-polish",
        action="store_true",
        help="Single-pass late-start polish; writes out-post-align.srt (default off)",
    )
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} on {args.device}…", flush=True)
    asr = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        tokenizer=args.processor,
        feature_extractor=args.processor,
        chunk_length_s=30,
        device=args.device,
        ignore_warning=True,
    )

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
        srt_path = ep_dir / "out.srt"
        txt_path = ep_dir / "out.txt"

        if args.skip_existing and srt_path.exists() and srt_path.stat().st_size > 500:
            print(f"SKIP existing {srt_path}")
            continue

        print(f"\n=== {src.name} → {ep_dir} ===", flush=True)
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

        est_min = duration_sec / 60 * cpu_minutes_per_audio_minute(args.model)
        print(f"  audio {duration_sec / 60:.1f} min — transcribing (CPU ~{est_min:.0f} min)…", flush=True)

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
                generate_kwargs=dict(WHISPER_GENERATE_KWARGS),
            )
            text = result.get("text", "")
            chunks = []
            for chunk in result.get("chunks") or []:
                t = (chunk.get("text") or "").strip()
                ts = chunk.get("timestamp")
                if not t or not ts or ts[0] is None:
                    continue
                end = ts[1] if len(ts) > 1 and ts[1] is not None else ts[0] + 2.0
                if not is_hallucination(t, end - ts[0]):
                    chunks.append({"timestamp": (ts[0], end), "text": t})
            if not chunks and text and not is_hallucination(text, duration_sec):
                chunks = [{"timestamp": (0.0, duration_sec), "text": text}]
            chunks = refine_chunks(chunks)
            text = "".join(c["text"] for c in chunks)

        if not verify_result(text, duration_sec, vad_stats.get("speech_sec")):
            print(f"  VERIFY FAILED ({len(text)} chars)", flush=True)
            failed.append(src.name)
            continue

        from inspect_alignment import EP1_OPEN_CHECKPOINTS

        checkpoints = EP1_OPEN_CHECKPOINTS if duration_sec <= 320 else []
        pre_srt = ep_dir / "out-pre-align.srt"
        write_srt(chunks, pre_srt)
        pre_report, pre_warnings = _write_alignment_reports(
            audio,
            pre_srt,
            ep_dir,
            duration_sec=duration_sec,
            report_name="alignment-pre.txt",
            issues_name="alignment-pre-issues.json",
            checkpoints=checkpoints,
        )
        print(f"  pre-align → {pre_srt} ({pre_warnings} alignment flags)", flush=True)

        polish_log: list[str] = []
        if args.align_polish:
            chunks, polish_log = finalize_alignment(audio, chunks, duration_sec)
            if polish_log:
                print(f"  align polish: {len(polish_log)} adjustment(s)", flush=True)
                for line in polish_log[:8]:
                    print(f"    {line}", flush=True)
                if len(polish_log) > 8:
                    print(f"    … +{len(polish_log) - 8} more", flush=True)
            post_srt = ep_dir / "out-post-align.srt"
            write_srt(chunks, post_srt)
            write_srt(chunks, srt_path)
            text = "".join(c["text"] for c in chunks)
            post_report, post_warnings = _write_alignment_reports(
                audio,
                post_srt,
                ep_dir,
                duration_sec=duration_sec,
                report_name="alignment.txt",
                issues_name="alignment_issues.json",
                checkpoints=checkpoints,
            )
            print(
                f"  post-align → {post_srt} ({post_warnings} alignment flags)",
                flush=True,
            )
            print(f"  out.srt = post-align copy", flush=True)
        else:
            write_srt(chunks, srt_path)
            post_warnings = pre_warnings
            (ep_dir / "alignment.txt").write_text(
                (ep_dir / "alignment-pre.txt").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (ep_dir / "alignment_issues.json").write_text(
                (ep_dir / "alignment-pre-issues.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            print(f"  out.srt = pre-align (polish off)", flush=True)

        txt_path.write_text(text.strip(), encoding="utf-8")

        if post_warnings:
            print(f"  ALIGN WARNINGS ({post_warnings}) → {ep_dir / 'alignment.txt'}", flush=True)
        else:
            print(f"  alignment OK → {ep_dir / 'alignment.txt'}", flush=True)

        meta = {
            "source": str(src),
            "duration_sec": duration_sec,
            "chunks": len(chunks),
            "chars": len(text),
            "model": args.model,
            "processor": args.processor,
            "vad": not args.no_vad,
            "align_polish": args.align_polish,
            **vad_stats,
            "alignment_pre_warnings": pre_warnings,
            "alignment_post_warnings": post_warnings if args.align_polish else pre_warnings,
            "alignment_polish_adjustments": len(polish_log),
        }
        (ep_dir / "out.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  OK → {srt_path} ({len(chunks)} cues, {len(text)} chars)", flush=True)
        if text:
            print(f"  preview: {text[:120]}…", flush=True)

    if failed:
        print("FAILED:", ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
