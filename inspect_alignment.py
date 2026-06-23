#!/usr/bin/env python3
"""Inspect and tighten subtitle timing vs audio energy (onset, offset, pauses)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_srt import Cue, parse_srt  # noqa: E402
from transcribe_cantonese import (  # noqa: E402
    MIN_CUE_SEC,
    MIN_GAP_SPEECH_RMS,
    MIN_OVERLAP_FIX_SEC,
    SAMPLE_RATE,
    load_audio,
    region_rms,
    srt_time,
)

# RMS frame analysis (40 ms window, 20 ms hop).
FRAME_SEC = 0.04
HOP_SEC = 0.02
CUE_SEARCH_PAD_SEC = 0.70
SPEECH_THRESH_FLOOR = MIN_GAP_SPEECH_RMS

# Warn thresholds (alignment report — tuned to reduce RMS false alarms on TV audio).
START_LATE_WARN_SEC = 0.45
END_EARLY_WARN_SEC = 0.40
START_EARLY_WARN_SEC = 0.70
END_LATE_WARN_SEC = 1.20
# Conservative polish: only pull clearly-late starts; never extend/shrink ends.
START_LATE_FIX_SEC = 0.35
MAX_POLISH_PULL_SEC = 0.55
MAX_POLISH_PASSES = 1

# Ep1 opening checkpoints (seconds) — quick QA for known trouble spots.
EP1_OPEN_CHECKPOINTS = [
    ("library", 74, 88),
    ("3:12", 191, 196),
    ("shirt", 238, 244),
    ("4:22", 260, 265),
    ("4:38", 276, 282),
    ("4:56", 294, 300),
]


@dataclass
class CueAlignment:
    index: int
    start: float
    end: float
    text: str
    speech_start: float | None
    speech_end: float | None
    start_delta: float | None  # cue.start - speech_start (+ = subtitle late)
    end_delta: float | None  # speech_end - cue.end (+ = subtitle ended early)
    issues: list[str]

    @property
    def fixable_start_late(self) -> float:
        if self.start_delta is None or self.start_delta < START_LATE_FIX_SEC:
            return 0.0
        return min(self.start_delta - 0.08, MAX_POLISH_PULL_SEC)

    @property
    def fixable_end_early(self) -> float:
        return 0.0

    @property
    def fixable_end_late(self) -> float:
        return 0.0


def _rms_frames(audio: np.ndarray, start_sec: float, end_sec: float) -> tuple[np.ndarray, np.ndarray]:
    s0 = max(0, int(start_sec * SAMPLE_RATE))
    s1 = min(len(audio), int(end_sec * SAMPLE_RATE))
    if s1 <= s0:
        return np.array([]), np.array([])
    frame = max(1, int(FRAME_SEC * SAMPLE_RATE))
    hop = max(1, int(HOP_SEC * SAMPLE_RATE))
    times: list[float] = []
    rms: list[float] = []
    pos = s0
    while pos + frame <= s1:
        chunk = audio[pos : pos + frame]
        times.append(pos / SAMPLE_RATE)
        rms.append(float(np.sqrt(np.mean(chunk * chunk))))
        pos += hop
    return np.array(times), np.array(rms)


def _sustained_threshold(rms: np.ndarray) -> float:
    if len(rms) == 0:
        return SPEECH_THRESH_FLOOR
    peak = float(np.max(rms))
    if peak < SPEECH_THRESH_FLOOR:
        return SPEECH_THRESH_FLOOR
    return max(peak * 0.26, SPEECH_THRESH_FLOOR, float(np.percentile(rms, 30)) * 1.35)


def _find_onset(times: np.ndarray, rms: np.ndarray, thresh: float) -> float | None:
    min_run = 3
    run = 0
    for i, v in enumerate(rms):
        if v >= thresh:
            run += 1
            if run >= min_run:
                return float(times[i - min_run + 1])
        else:
            run = 0
    return None


def _find_offset(times: np.ndarray, rms: np.ndarray, thresh: float) -> float | None:
    min_run = 3
    run = 0
    for i in range(len(rms) - 1, -1, -1):
        if rms[i] >= thresh:
            run += 1
            if run >= min_run:
                idx = i + min_run - 1
                return float(times[min(idx, len(times) - 1)] + FRAME_SEC)
        else:
            run = 0
    return None


def speech_bounds_for_cue(
    audio: np.ndarray,
    cue_start: float,
    cue_end: float,
    *,
    duration_sec: float,
) -> tuple[float | None, float | None]:
    """Estimate speech onset near cue start and offset near cue end."""
    onset_win = (max(0.0, cue_start - CUE_SEARCH_PAD_SEC), cue_start + 1.25)
    offset_win = (cue_end - 1.25, min(duration_sec, cue_end + 0.35))

    t_on, rms_on = _rms_frames(audio, *onset_win)
    t_off, rms_off = _rms_frames(audio, *offset_win)

    speech_start = None
    speech_end = None
    if len(rms_on):
        speech_start = _find_onset(t_on, rms_on, _sustained_threshold(rms_on))
    if len(rms_off):
        speech_end = _find_offset(t_off, rms_off, _sustained_threshold(rms_off))

    return speech_start, speech_end


def analyze_cue_alignment(
    audio: np.ndarray,
    cue: Cue | dict,
    *,
    index: int,
    duration_sec: float,
) -> CueAlignment:
    if isinstance(cue, dict):
        start, end = cue["timestamp"]
        text = cue.get("text", "")
    else:
        start, end = cue.start, cue.end
        text = cue.text

    speech_start, speech_end = speech_bounds_for_cue(
        audio, start, end, duration_sec=duration_sec
    )
    start_delta = (start - speech_start) if speech_start is not None else None
    end_delta = (speech_end - end) if speech_end is not None else None
    issues: list[str] = []

    if start_delta is not None:
        if start_delta >= START_LATE_WARN_SEC:
            issues.append(f"LATE-START +{start_delta:.2f}s")
        elif start_delta <= -START_EARLY_WARN_SEC and (end - start) < 5.0:
            issues.append(f"EARLY-START {start_delta:.2f}s")
    if end_delta is not None:
        if end_delta >= END_EARLY_WARN_SEC:
            tail_rms = region_rms(audio, end, min(duration_sec, end + 0.40))
            if tail_rms >= SPEECH_THRESH_FLOOR * 0.75:
                issues.append(f"EARLY-END -{end_delta:.2f}s")
        elif end_delta <= -END_LATE_WARN_SEC and (end - start) < 5.0:
            issues.append(f"LINGER +{-end_delta:.2f}s")

    return CueAlignment(
        index=index,
        start=start,
        end=end,
        text=text,
        speech_start=speech_start,
        speech_end=speech_end,
        start_delta=start_delta,
        end_delta=end_delta,
        issues=issues,
    )


def analyze_all_cues(
    audio: np.ndarray,
    cues: list[Cue] | list[dict],
    *,
    duration_sec: float | None = None,
) -> list[CueAlignment]:
    duration_sec = duration_sec or len(audio) / SAMPLE_RATE
    out: list[CueAlignment] = []
    for i, cue in enumerate(cues, start=1):
        out.append(analyze_cue_alignment(audio, cue, index=i, duration_sec=duration_sec))
    return out


def adjust_chunks_to_audio(
    audio: np.ndarray,
    chunks: list[dict],
    *,
    duration_sec: float | None = None,
) -> tuple[list[dict], list[str]]:
    """Conservative polish: pull late starts only; respect neighbours (no end edits)."""
    if not chunks:
        return chunks, []
    duration_sec = duration_sec or len(audio) / SAMPLE_RATE
    sorted_in = sorted(chunks, key=lambda c: c["timestamp"][0])
    analyses = analyze_all_cues(audio, sorted_in, duration_sec=duration_sec)
    adjusted: list[dict] = []
    log: list[str] = []

    for i, (chunk, analysis) in enumerate(zip(sorted_in, analyses)):
        start, end = chunk["timestamp"]
        text = chunk["text"]
        new_start = start
        prev_end = adjusted[-1]["timestamp"][1] if adjusted else 0.0
        next_start = (
            sorted_in[i + 1]["timestamp"][0] if i + 1 < len(sorted_in) else duration_sec
        )

        pull = analysis.fixable_start_late
        if pull > 0:
            candidate = max(0.0, start - pull)
            candidate = max(candidate, prev_end + MIN_OVERLAP_FIX_SEC)
            if candidate + MIN_CUE_SEC <= next_start:
                if candidate < start - 0.02:
                    new_start = candidate
                    log.append(
                        f"fix cue {analysis.index} start {srt_time(start)}→{srt_time(new_start)} "
                        f"(late +{analysis.start_delta:.2f}s)"
                    )

        adjusted.append({"timestamp": (new_start, end), "text": text})

    return adjusted, log


def polish_chunks_alignment(
    audio: np.ndarray,
    chunks: list[dict],
    *,
    duration_sec: float | None = None,
    max_passes: int = MAX_POLISH_PASSES,
) -> tuple[list[dict], list[str]]:
    """Repeat timing fixes until no fixable drift remains (or max passes)."""
    if not chunks:
        return chunks, []
    duration_sec = duration_sec or len(audio) / SAMPLE_RATE
    all_log: list[str] = []
    current = chunks
    for _ in range(max_passes):
        current, log = adjust_chunks_to_audio(audio, current, duration_sec=duration_sec)
        if not log:
            break
        all_log.extend(log)
    return current, all_log


def chunks_to_cues(chunks: list[dict]) -> list[Cue]:
    return [Cue(c["timestamp"][0], c["timestamp"][1], c["text"]) for c in chunks]


def find_untranscribed_gaps(
    audio: np.ndarray,
    cues: list[Cue],
    *,
    min_gap_sec: float = 0.45,
) -> list[tuple[float, float, float]]:
    """Return (gap_start, gap_end, rms) for gaps that may contain speech."""
    gaps: list[tuple[float, float, float]] = []
    for i in range(len(cues) - 1):
        gap_start = cues[i].end
        gap_end = cues[i + 1].start
        gap_dur = gap_end - gap_start
        if gap_dur < min_gap_sec:
            continue
        rms = region_rms(audio, gap_start, gap_end)
        min_rms = MIN_GAP_SPEECH_RMS if gap_dur < 25.0 else MIN_GAP_SPEECH_RMS * 1.35
        if rms >= min_rms:
            gaps.append((gap_start, gap_end, rms))
    return gaps


def inspect_alignment(
    audio: np.ndarray,
    cues: list[Cue],
    *,
    duration_sec: float | None = None,
    checkpoints: list[tuple[str, float, float]] | None = None,
) -> tuple[str, list[str]]:
    """Return (report text, list of warning lines)."""
    duration_sec = duration_sec or len(audio) / SAMPLE_RATE
    warnings: list[str] = []
    analyses = analyze_all_cues(audio, cues, duration_sec=duration_sec)

    late_starts = [a for a in analyses if any(i.startswith("LATE-START") for i in a.issues)]
    early_ends = [a for a in analyses if any(i.startswith("EARLY-END") for i in a.issues)]
    lingers = [a for a in analyses if any(i.startswith("LINGER") for i in a.issues)]
    overlaps = 0
    zero_gaps = 0

    lines = [
        f"=== Alignment proofread ({len(cues)} cues, {duration_sec / 60:.1f} min) ===",
        "",
        "Summary:",
        f"  LATE-START: {len(late_starts)}",
        f"  EARLY-END: {len(early_ends)}",
        f"  LINGER: {len(lingers)}",
        "",
    ]

    timing_residual = [a for a in analyses if any(
        i.startswith(("LATE-START", "EARLY-END", "LINGER", "EARLY-START"))
        for i in a.issues
    )]
    for a in timing_residual:
        sp = (
            f"{srt_time(a.speech_start)}–{srt_time(a.speech_end)}"
            if a.speech_start is not None and a.speech_end is not None
            else "no speech detected"
        )
        msg = (
            f"cue {a.index} @ {srt_time(a.start)}–{srt_time(a.end)} "
            f"audio {sp} | {', '.join(a.issues)} | '{a.text[:28]}'"
        )
        warnings.append(msg)
        lines.append(f"  ! {msg}")

    if not timing_residual:
        lines.append("  Per-cue timing: clean (no residual drift).")
    lines.append("")

    lines.append("Cue spacing:")
    for i in range(len(cues) - 1):
        gap = cues[i + 1].start - cues[i].end
        if gap < -0.02:
            overlaps += 1
            msg = (
                f"OVERLAP {-gap:.2f}s cue {i + 1}→{i + 2} @ {srt_time(cues[i].end)}: "
                f"'{cues[i].text[-18:]}' / '{cues[i + 1].text[:18]}'"
            )
            warnings.append(msg)
            lines.append(f"  ! {msg}")
        elif gap < 0.03:
            zero_gaps += 1
            msg = (
                f"ZERO-GAP cue {i + 1}→{i + 2} @ {srt_time(cues[i].end)}: "
                f"'{cues[i].text[-18:]}' / '{cues[i + 1].text[:18]}'"
            )
            warnings.append(msg)
            lines.append(f"  ! {msg}")

    lines.append("")
    lines.append("Untranscribed gaps (speech energy between cues):")
    gap_hits = find_untranscribed_gaps(audio, cues)
    if not gap_hits:
        lines.append("  none flagged")
    for gap_start, gap_end, rms in gap_hits:
        gap_dur = gap_end - gap_start
        msg = (
            f"UNTRANSCRIBED gap {gap_dur:.1f}s @ {srt_time(gap_start)}–{srt_time(gap_end)} "
            f"rms={rms:.4f}"
        )
        warnings.append(msg)
        lines.append(f"  ? {msg}")

    cps = checkpoints if checkpoints is not None else []
    if cps:
        lines.append("")
        lines.append("Checkpoint coverage:")
        for label, t0, t1 in cps:
            in_win = [c for c in cues if c.end > t0 and c.start < t1]
            rms = region_rms(audio, t0, t1)
            status = f"{len(in_win)} cue(s)" if in_win else "NO CUES"
            lines.append(f"  {label} ({srt_time(t0)}–{srt_time(t1)}) rms={rms:.4f}: {status}")
            if not in_win and rms >= MIN_GAP_SPEECH_RMS:
                warnings.append(f"CHECKPOINT MISS {label} rms={rms:.4f}")

    lines.append("")
    lines.append(
        f"Warnings: {len(warnings)} "
        f"(timing={len(timing_residual)}, overlap={overlaps}, zero-gap={zero_gaps}, "
        f"gaps={len(gap_hits)})"
    )
    if warnings:
        lines.append("Spot-check flagged cues against the video.")
    else:
        lines.append("No alignment warnings.")
    return "\n".join(lines), warnings


def alignment_issues_json(analyses: list[CueAlignment]) -> list[dict]:
    return [
        asdict(a)
        for a in analyses
        if a.issues or a.speech_start is not None
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect SRT alignment vs audio")
    parser.add_argument("srt", type=Path)
    parser.add_argument("audio", type=Path, help="WAV or video (needs ffmpeg)")
    parser.add_argument("-o", "--report-out", type=Path)
    parser.add_argument("--json-out", type=Path, help="Per-cue alignment details (JSON)")
    args = parser.parse_args()

    if not args.srt.is_file():
        print(f"Not found: {args.srt}", file=sys.stderr)
        return 1

    cues = parse_srt(args.srt)
    if args.audio.suffix.lower() in {".wav", ".flac"}:
        audio, dur = load_audio(args.audio)
    else:
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "a.wav"
            subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-y", "-i", str(args.audio),
                    "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), str(wav),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            audio, dur = load_audio(wav)

    report, warnings = inspect_alignment(
        audio, cues, duration_sec=dur, checkpoints=EP1_OPEN_CHECKPOINTS
    )
    print(report)
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(report, encoding="utf-8")
        print(f"Report → {args.report_out}", file=sys.stderr)
    if args.json_out:
        analyses = analyze_all_cues(audio, cues, duration_sec=dur)
        args.json_out.write_text(
            json.dumps(alignment_issues_json(analyses), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON → {args.json_out}", file=sys.stderr)
    return 1 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
