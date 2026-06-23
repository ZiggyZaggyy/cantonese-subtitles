#!/usr/bin/env python3
"""Compare and merge Cantonese SRT outputs (VAD vs legacy)."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Reuse hallucination checks from transcribe_cantonese
sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcribe_cantonese import is_hallucination, srt_time, write_srt  # noqa: E402


@dataclass
class Cue:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def parse_time(ts: str) -> float:
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(path: Path) -> list[Cue]:
    content = path.read_text(encoding="utf-8")
    cues: list[Cue] = []
    for block in re.split(r"\n\n+", content.strip()):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        m = re.match(r"(.+?)\s*-->\s*(.+)", lines[1])
        if not m:
            continue
        start = parse_time(m.group(1).strip())
        end = parse_time(m.group(2).strip())
        text = "\n".join(lines[2:]).strip()
        cues.append(Cue(start, end, text))
    return cues


def overlaps(a: Cue, b: Cue, slack: float = 1.0) -> bool:
    return a.start < b.end + slack and b.start < a.end + slack


def filter_clean(cues: list[Cue]) -> list[Cue]:
    out: list[Cue] = []
    for c in cues:
        if is_hallucination(c.text, max(c.duration, 0.5)):
            continue
        out.append(c)
    return out


def merge_cues(primary: list[Cue], secondary: list[Cue], *, slack: float = 1.5) -> list[Cue]:
    """Prefer primary (VAD); add secondary only where primary has no overlap."""
    merged = list(primary)
    for sc in secondary:
        if any(overlaps(sc, pc, slack) for pc in primary):
            continue
        merged.append(sc)
    merged.sort(key=lambda c: c.start)
    return merged


def cues_to_chunks(cues: list[Cue]) -> list[dict]:
    return [{"timestamp": (c.start, c.end), "text": c.text} for c in cues]


def report(
    label_a: str,
    cues_a: list[Cue],
    label_b: str,
    cues_b: list[Cue],
    *,
    t0: float = 0.0,
    t1: float | None = None,
) -> str:
    t1 = t1 if t1 is not None else max((c.end for c in cues_a + cues_b), default=t0)
    a = [c for c in cues_a if c.end >= t0 and c.start <= t1]
    b = [c for c in cues_b if c.end >= t0 and c.start <= t1]
    lines = [
        f"=== {label_a} ({len(a)} cues) vs {label_b} ({len(b)} cues) "
        f"[{srt_time(t0)} – {srt_time(t1)}] ===",
        "",
    ]
    ai, bi = 0, 0
    while ai < len(a) or bi < len(b):
        ca = a[ai] if ai < len(a) else None
        cb = b[bi] if bi < len(b) else None
        if ca and cb:
            t = min(ca.start, cb.start)
        elif ca:
            t = ca.start
        else:
            t = cb.start  # type: ignore[union-attr]
        lines.append(f"@{srt_time(t)}")
        if ca and (not cb or ca.start <= cb.start + 0.5):
            flag = "HALLUC" if is_hallucination(ca.text, ca.duration) else "ok"
            lines.append(f"  {label_a:8} [{srt_time(ca.start)}→{srt_time(ca.end)}] ({flag}) {ca.text[:90]}")
            ai += 1
        elif cb:
            flag = "HALLUC" if is_hallucination(cb.text, cb.duration) else "ok"
            lines.append(f"  {label_b:8} [{srt_time(cb.start)}→{srt_time(cb.end)}] ({flag}) {cb.text[:90]}")
            bi += 1
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare / merge SRT files")
    parser.add_argument("vad_srt", type=Path, help="New VAD-based SRT")
    parser.add_argument("legacy_srt", type=Path, help="Original / legacy SRT")
    parser.add_argument("--from-sec", type=float, default=0.0)
    parser.add_argument("--to-sec", type=float, default=0.0, help="0 = end of files")
    parser.add_argument("--merge-out", type=Path, help="Write merged SRT (VAD + clean legacy gaps)")
    parser.add_argument("--report-out", type=Path, help="Write side-by-side report")
    args = parser.parse_args()

    vad = parse_srt(args.vad_srt)
    legacy = parse_srt(args.legacy_srt)
    legacy_clean = filter_clean(legacy)
    t1 = args.to_sec if args.to_sec > 0 else None

    text = report("VAD", vad, "LEGACY", legacy, t0=args.from_sec, t1=t1)
    print(text)

    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(text, encoding="utf-8")
        print(f"Report → {args.report_out}", file=sys.stderr)

    if args.merge_out:
        merged = merge_cues(vad, legacy_clean)
        if t1 is not None:
            merged = [c for c in merged if c.end >= args.from_sec and c.start <= t1]
        write_srt(cues_to_chunks(merged), args.merge_out)
        print(f"Merged {len(merged)} cues → {args.merge_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
