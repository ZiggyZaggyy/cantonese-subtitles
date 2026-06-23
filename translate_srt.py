#!/usr/bin/env python3
"""Optional: Cantonese SRT → fluent Mandarin via OpenAI (ChatGPT-quality).

Machine translators (mT5, Google) only do word-for-word transliteration and are
not used here. This script requires OPENAI_API_KEY and calls the Chat API with
a subtitle-localization prompt.

Not run automatically by iterate-ep1.sh — use only when you want Mandarin subs:

  export OPENAI_API_KEY=sk-...
  python translate_srt.py out.srt -o out-zh.srt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_srt import parse_srt  # noqa: E402
from transcribe_cantonese import is_hallucination, write_srt  # noqa: E402

DEFAULT_MODEL = "gpt-4o-mini"
BATCH_SIZE = 20

SYSTEM_PROMPT = """You translate Cantonese TV drama subtitles into natural, fluent Mandarin (Simplified Chinese) for viewers who do NOT speak Cantonese.

Requirements:
- Rewrite to idiomatic spoken Mandarin — never word-for-word Cantonese transliteration
- Convert Cantonese grammar and particles to Mandarin (嘅→的, 唔→不, 喺→在, 佢→他/她, 咩→什么, 喇/啦→了/啊 as appropriate)
- One output string per input line, same count and order as input
- Keep character names unchanged
- Split long lines with a newline only if the input had multiple sentences that need separate subtitle lines
- Brief parenthetical gloss only when a term is obscure (e.g. 水鱼 → 冤大头)

Cantonese terms (use natural Mandarin, gloss if helpful):
- 老姑婆 → 老处女/老小姐（调侃）
- 耳目一新 / 一身耳目 → 耳目一新
- 倔 → 固执、倔强
- 水鱼 → 冤大头
- 飞仔 → 小混混
- 核突 → 难看、丑
- 保龄头 → 保龄球头

Return ONLY valid JSON: {"lines": ["...", ...]} with exactly the same number of lines as input."""


def translate_batch_openai(texts: list[str], *, model: str) -> list[str]:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in the environment")

    client = OpenAI(api_key=api_key)
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    user = (
        f"Translate these {len(texts)} Cantonese subtitle lines to fluent Mandarin.\n\n"
        f"{numbered}"
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    data = json.loads(raw)
    lines = data.get("lines") or data.get("translations") or data.get("output")
    if not isinstance(lines, list):
        raise ValueError(f"Unexpected API response shape: {raw[:200]}")
    if len(lines) != len(texts):
        raise ValueError(f"Expected {len(texts)} lines, got {len(lines)}")
    return [str(x).strip() for x in lines]


def translate_openai(texts: list[str], *, model: str, batch_size: int) -> list[str]:
    out: list[str] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        out.extend(translate_batch_openai(batch, model=model))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cantonese SRT → fluent Mandarin via OpenAI (optional; needs API key)",
    )
    parser.add_argument("input_srt", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="Default: <input>-zh.srt")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--skip-hallucinations",
        action="store_true",
        help="Drop cues matching streaming-outro / repetition filters",
    )
    args = parser.parse_args()

    src = args.input_srt
    if not src.is_file():
        print(f"Not found: {src}", file=sys.stderr)
        return 1

    out = args.output or src.with_name(src.stem + "-zh.srt")
    cues = parse_srt(src)
    if not cues:
        print(f"No cues in {src}", file=sys.stderr)
        return 1

    kept = []
    texts: list[str] = []
    for cue in cues:
        dur = max(cue.end - cue.start, 0.5)
        if args.skip_hallucinations and is_hallucination(cue.text, dur):
            print(f"  skip hallucination: {cue.text[:40]!r}", flush=True)
            continue
        kept.append(cue)
        texts.append(cue.text.replace("\n", " ").strip())

    if not kept:
        print("No cues left after filtering", file=sys.stderr)
        return 1

    print(f"Translating {len(kept)} cues via OpenAI ({args.model})…", flush=True)
    try:
        translated = translate_openai(texts, model=args.model, batch_size=args.batch_size)
    except Exception as exc:
        print(f"Translation failed: {exc}", file=sys.stderr)
        return 1

    zh_chunks = []
    for cue, zh in zip(kept, translated):
        zh_text = re.sub(r"\s+", " ", (zh or "").strip())
        zh_chunks.append({"timestamp": (cue.start, cue.end), "text": zh_text})

    write_srt(zh_chunks, out)
    meta = {
        "source_srt": str(src),
        "output_srt": str(out),
        "cues": len(zh_chunks),
        "translator": f"openai:{args.model}",
        "skipped_hallucinations": len(cues) - len(kept),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK → {out} ({len(zh_chunks)} cues)", flush=True)
    if zh_chunks:
        print(f"  preview: {zh_chunks[0]['text'][:80]}…", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
