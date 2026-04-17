"""HTML reporter — static, side-by-side R003, shingle highlights."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from rag_lint.models import Document, Finding, Severity
from rag_lint.reporters.shingles import highlight_html, shingle_highlights

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def render_html(
    documents: list[Document],
    findings: list[Finding],
    output: Path,
    corpus_path: Path,
) -> None:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("report.html.j2")

    sorted_findings = sorted(findings, key=lambda f: f.sort_key())
    doc_by_path = {d.rel_path: d for d in documents}
    findings_ctx = [_finding_context(f, doc_by_path) for f in sorted_findings]

    class_counts = Counter(
        d.classification.value if d.classification else "(missing)" for d in documents
    )
    totals = {
        "total": len(findings),
        "high": sum(1 for f in findings if f.severity == Severity.HIGH),
        "medium": sum(1 for f in findings if f.severity == Severity.MEDIUM),
        "low": sum(1 for f in findings if f.severity == Severity.LOW),
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html = template.render(
        corpus_path=str(corpus_path),
        corpus_name=corpus_path.name,
        timestamp=timestamp,
        doc_count=len(documents),
        para_count=sum(len(d.paragraphs) for d in documents),
        class_counts=dict(class_counts),
        totals=totals,
        findings=findings_ctx,
    )
    output.write_text(html, encoding="utf-8")


def _finding_context(f: Finding, doc_by_path: dict[str, Document]) -> dict:
    ctx: dict = {
        "rule_id": f.rule_id,
        "rule_name": f.rule_name,
        "severity": f.severity.value,
        "primary_file": f.primary_file,
        "related_files": f.related_files,
        "detail": f.detail,
        "fix": f.fix,
        "score": f.score,
        "kind": f.rule_id,
    }

    if f.rule_id == "R002":
        primary_doc = doc_by_path[f.evidence["primary_path"]]
        other_doc = doc_by_path[f.evidence["other_path"]]
        pairs = []
        for p_idx, o_idx, sim in f.evidence["matched_pairs"][:6]:
            p_text = primary_doc.paragraphs[p_idx].text
            o_text = other_doc.paragraphs[o_idx].text
            p_spans, o_spans = shingle_highlights(p_text, o_text)
            pairs.append(
                {
                    "primary_html": highlight_html(p_text, p_spans),
                    "other_html": highlight_html(o_text, o_spans),
                    "primary_line": primary_doc.paragraphs[p_idx].start_line,
                    "other_line": other_doc.paragraphs[o_idx].start_line,
                    "cosine": sim,
                }
            )
        ctx["pairs"] = pairs
        ctx["pair_count"] = len(f.evidence["matched_pairs"])
        ctx["primary_classification"] = f.evidence["primary_classification"]
        ctx["other_classification"] = f.evidence["other_classification"]
        ctx["primary_path"] = f.evidence["primary_path"]
        ctx["other_path"] = f.evidence["other_path"]
        ctx["primary_ratio"] = f.evidence["primary_ratio"]
        ctx["other_ratio"] = f.evidence["other_ratio"]

    elif f.rule_id == "R003":
        a_doc = doc_by_path[f.evidence["a_path"]]
        b_doc = doc_by_path[f.evidence["b_path"]]
        higher_path = f.evidence["higher_path"]
        pairs = []
        for a_idx, b_idx, sim in f.evidence["all_pairs"][:4]:
            a_text = a_doc.paragraphs[a_idx].text
            b_text = b_doc.paragraphs[b_idx].text
            a_spans, b_spans = shingle_highlights(a_text, b_text)
            a_is_higher = a_doc.rel_path == higher_path
            higher_text = a_text if a_is_higher else b_text
            higher_spans = a_spans if a_is_higher else b_spans
            lower_text = b_text if a_is_higher else a_text
            lower_spans = b_spans if a_is_higher else a_spans
            higher_line = (
                a_doc.paragraphs[a_idx].start_line
                if a_is_higher
                else b_doc.paragraphs[b_idx].start_line
            )
            lower_line = (
                b_doc.paragraphs[b_idx].start_line
                if a_is_higher
                else a_doc.paragraphs[a_idx].start_line
            )
            pairs.append(
                {
                    "higher_html": highlight_html(higher_text, higher_spans),
                    "lower_html": highlight_html(lower_text, lower_spans),
                    "higher_line": higher_line,
                    "lower_line": lower_line,
                    "cosine": sim,
                }
            )
        ctx["pairs"] = pairs
        ctx["higher_classification"] = f.evidence["higher_classification"]
        ctx["lower_classification"] = f.evidence["lower_classification"]
        ctx["higher_path"] = f.evidence["higher_path"]
        ctx["lower_path"] = f.evidence["lower_path"]
        ctx["max_cosine"] = f.evidence["max_cosine"]

    return ctx
