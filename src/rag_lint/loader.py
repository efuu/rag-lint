"""Corpus loader: discovers .md files, parses YAML frontmatter, builds Documents."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from rag_lint.chunking import chunk_paragraphs
from rag_lint.models import Classification, Document

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)", re.DOTALL)


def load_corpus(root: Path) -> list[Document]:
    """Load all *.md files under root (recursive). Order deterministic by rel_path."""
    paths = sorted(p for p in root.rglob("*.md") if p.is_file())
    docs = [load_document(p, root) for p in paths]
    return docs


def load_document(path: Path, root: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    meta, body, body_start_line = _split_frontmatter(raw)

    classification = None
    cls_raw = meta.get("classification")
    if isinstance(cls_raw, str):
        try:
            classification = Classification(cls_raw.strip().lower())
        except ValueError:
            classification = None

    doc_type = meta.get("doc_type") if isinstance(meta.get("doc_type"), str) else None
    date_raw = meta.get("date")
    date = str(date_raw) if date_raw is not None else None

    paragraphs = chunk_paragraphs(body, body_start_line=body_start_line)

    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = path.name

    return Document(
        path=path,
        rel_path=rel,
        classification=classification,
        doc_type=doc_type,
        date=date,
        raw_text=raw,
        body=body,
        paragraphs=paragraphs,
    )


def _split_frontmatter(raw: str) -> tuple[dict, str, int]:
    m = _FRONTMATTER.match(raw)
    if not m:
        return {}, raw, 1

    yaml_block = m.group(1)
    body = m.group(2)
    try:
        meta = yaml.safe_load(yaml_block) or {}
    except yaml.YAMLError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}

    consumed = m.end(2) - len(body)
    body_start_line = raw.count("\n", 0, consumed) + 1
    return meta, body, body_start_line
