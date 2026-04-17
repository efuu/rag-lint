"""Paragraph chunking: split document body on blank lines."""

from __future__ import annotations

import re

from rag_lint.models import Paragraph

_HEADER_LINE = re.compile(r"^\s{0,3}#{1,6}\s")


def chunk_paragraphs(body: str, body_start_line: int) -> list[Paragraph]:
    """Split body text on blank lines. Drop pure-whitespace and pure-header chunks.

    body_start_line is the 1-indexed line number where body begins in the file.
    """
    lines = body.splitlines()
    paragraphs: list[Paragraph] = []
    buf: list[str] = []
    buf_start = body_start_line
    idx = 0

    for offset, line in enumerate(lines):
        if line.strip() == "":
            if buf:
                text = "\n".join(buf).strip()
                if text and not _all_headers(buf):
                    paragraphs.append(
                        Paragraph(index=idx, text=text, start_line=buf_start)
                    )
                    idx += 1
                buf = []
            buf_start = body_start_line + offset + 1
        else:
            if not buf:
                buf_start = body_start_line + offset
            buf.append(line)

    if buf:
        text = "\n".join(buf).strip()
        if text and not _all_headers(buf):
            paragraphs.append(Paragraph(index=idx, text=text, start_line=buf_start))

    return paragraphs


def _all_headers(lines: list[str]) -> bool:
    stripped = [ln for ln in lines if ln.strip()]
    if not stripped:
        return True
    return all(_HEADER_LINE.match(ln) for ln in stripped)
