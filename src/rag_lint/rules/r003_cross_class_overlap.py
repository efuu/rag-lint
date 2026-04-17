"""R003 — cross-classification-semantic-overlap."""

from __future__ import annotations

import numpy as np

from rag_lint.models import CLASSIFICATION_RANK, Document, Finding, Severity
from rag_lint.rules.base import Rule

OVERLAP_COSINE = 0.75


class R003(Rule):
    id = "R003"
    name = "cross-classification-semantic-overlap"
    severity = Severity.HIGH

    def check(
        self, documents: list[Document], prior_findings: list[Finding]
    ) -> list[Finding]:
        suppressed_pairs = _r002_pairs(prior_findings)
        findings: list[Finding] = []
        classified = [d for d in documents if d.classification is not None]

        for i in range(len(classified)):
            for j in range(i + 1, len(classified)):
                a, b = classified[i], classified[j]
                if a.classification == b.classification:
                    continue
                pair_key = frozenset({a.rel_path, b.rel_path})
                if pair_key in suppressed_pairs:
                    continue
                f = _check_pair(a, b)
                if f is not None:
                    findings.append(f)
        return findings


def _r002_pairs(prior_findings: list[Finding]) -> set[frozenset[str]]:
    out: set[frozenset[str]] = set()
    for f in prior_findings:
        if f.rule_id == "R002":
            pair = frozenset({f.primary_file, *f.related_files})
            out.add(pair)
    return out


def _check_pair(a: Document, b: Document) -> Finding | None:
    a_vecs = _stack(a)
    b_vecs = _stack(b)
    if a_vecs.size == 0 or b_vecs.size == 0:
        return None

    sim = a_vecs @ b_vecs.T
    i_flat = int(sim.argmax())
    i, j = np.unravel_index(i_flat, sim.shape)
    max_cos = float(sim[i, j])

    if max_cos < OVERLAP_COSINE:
        return None

    higher, lower, h_idx, l_idx = _order_by_class(a, b, int(i), int(j))

    # Collect all pair matches above threshold for evidence.
    hits = np.argwhere(sim >= OVERLAP_COSINE)
    all_pairs: list[tuple[int, int, float]] = []
    for ii, jj in hits:
        all_pairs.append((int(ii), int(jj), float(sim[ii, jj])))
    all_pairs.sort(key=lambda t: -t[2])

    detail = (
        f"Paragraph in `{higher.rel_path}` ({higher.classification.value}) has cosine {max_cos:.2f} "
        f"to a paragraph in `{lower.rel_path}` ({lower.classification.value}); "
        f"overlap creates the precondition for a paraphrase leak across classifications."
    )
    fix = (
        f"Redact the overlapping section from `{higher.rel_path}` or rewrite so the shared framing "
        f"no longer appears in both documents."
    )
    return Finding(
        rule_id="R003",
        rule_name="cross-classification-semantic-overlap",
        severity=Severity.HIGH,
        primary_file=higher.rel_path,
        related_files=[lower.rel_path],
        detail=detail,
        fix=fix,
        score=max_cos,
        evidence={
            "higher_path": higher.rel_path,
            "lower_path": lower.rel_path,
            "higher_classification": higher.classification.value,
            "lower_classification": lower.classification.value,
            "max_cosine": max_cos,
            "higher_para_index": h_idx,
            "lower_para_index": l_idx,
            "all_pairs": all_pairs,  # list[(a_idx, b_idx, cos)] in original (a,b) orientation
            "a_path": a.rel_path,
            "b_path": b.rel_path,
        },
    )


def _order_by_class(
    a: Document, b: Document, a_idx: int, b_idx: int
) -> tuple[Document, Document, int, int]:
    a_rank = CLASSIFICATION_RANK[a.classification]
    b_rank = CLASSIFICATION_RANK[b.classification]
    if a_rank >= b_rank:
        return a, b, a_idx, b_idx
    return b, a, b_idx, a_idx


def _stack(doc: Document) -> np.ndarray:
    if not doc.paragraphs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack([p.embedding for p in doc.paragraphs])
