"""stdout reporter."""

from __future__ import annotations

from typing import TextIO

from rag_lint.models import Document, Finding, Severity


def render_stdout(
    documents: list[Document], findings: list[Finding], stream: TextIO
) -> int:
    """Render findings to stream. Return exit code (0 clean, 1 findings present)."""
    sorted_findings = sorted(findings, key=lambda f: f.sort_key())

    for f in sorted_findings:
        stream.write(f"{f.primary_file}: {f.rule_id}\n")
        stream.write(f"  {f.rule_name} [severity: {f.severity.value}]\n")
        stream.write(f"  {f.detail}\n")
        if f.related_files:
            stream.write(f"  related: {', '.join(f.related_files)}\n")
        stream.write(f"  Fix: {f.fix}\n")
        stream.write("\n")

    stream.write(_summary(documents, sorted_findings))
    stream.write("\n")
    return 0 if not findings else 1


def _summary(documents: list[Document], findings: list[Finding]) -> str:
    high = sum(1 for f in findings if f.severity == Severity.HIGH)
    med = sum(1 for f in findings if f.severity == Severity.MEDIUM)
    low = sum(1 for f in findings if f.severity == Severity.LOW)
    total = len(findings)
    doc_count = len(documents)
    para_count = sum(len(d.paragraphs) for d in documents)
    verdict = "clean" if total == 0 else f"{total} finding{'s' if total != 1 else ''}"
    return (
        f"---\n"
        f"Summary: {doc_count} documents, {para_count} paragraphs scanned. "
        f"{verdict} ({high} high, {med} medium, {low} low)."
    )
