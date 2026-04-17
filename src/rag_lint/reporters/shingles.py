"""4-gram shingle highlighter for side-by-side paragraph views."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")

Span = tuple[int, int]  # (start_char, end_char)
SHINGLE_N = 4


def shingle_highlights(text_a: str, text_b: str) -> tuple[list[Span], list[Span]]:
    """Return merged highlight spans for text_a and text_b based on shared 4-grams."""
    tokens_a = _tokens(text_a)
    tokens_b = _tokens(text_b)

    grams_a = _ngrams(tokens_a, SHINGLE_N)
    grams_b = _ngrams(tokens_b, SHINGLE_N)

    keys_a = {k for k, _ in grams_a}
    keys_b = {k for k, _ in grams_b}
    shared = keys_a & keys_b

    spans_a = _merge([s for k, s in grams_a if k in shared])
    spans_b = _merge([s for k, s in grams_b if k in shared])
    return spans_a, spans_b


def _tokens(text: str) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    for m in _TOKEN.finditer(text):
        out.append((m.group(0).lower(), m.start(), m.end()))
    return out


def _ngrams(tokens: list[tuple[str, int, int]], n: int) -> list[tuple[str, Span]]:
    out: list[tuple[str, Span]] = []
    for i in range(len(tokens) - n + 1):
        window = tokens[i : i + n]
        key = " ".join(t[0] for t in window)
        span = (window[0][1], window[-1][2])
        out.append((key, span))
    return out


def _merge(spans: list[Span]) -> list[Span]:
    if not spans:
        return []
    spans = sorted(spans)
    merged: list[Span] = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def highlight_html(text: str, spans: list[Span]) -> str:
    """Render text as HTML with <mark> around span regions. Escapes rest."""
    from html import escape

    if not spans:
        return escape(text)
    out: list[str] = []
    cursor = 0
    for s, e in spans:
        if s > cursor:
            out.append(escape(text[cursor:s]))
        out.append(f'<mark class="shingle">{escape(text[s:e])}</mark>')
        cursor = e
    if cursor < len(text):
        out.append(escape(text[cursor:]))
    return "".join(out)
