"""R002 — near-duplicate-across-classifications."""

from __future__ import annotations

import numpy as np

from rag_lint.models import Document, Finding, Severity
from rag_lint.rules.base import Rule

NEAR_DUP_COSINE = 0.90
NEAR_DUP_PROPORTION = 0.30


class R002(Rule):
    id = "R002"
    name = "near-duplicate-across-classifications"
    severity = Severity.MEDIUM

    def check(
        self, documents: list[Document], prior_findings: list[Finding]
    ) -> list[Finding]:
        findings: list[Finding] = []
        classified = [d for d in documents if d.classification is not None]

        for i in range(len(classified)):
            for j in range(i + 1, len(classified)):
                a, b = classified[i], classified[j]
                if a.classification == b.classification:
                    continue
                f = _check_pair(a, b)
                if f is not None:
                    findings.append(f)
        return findings


def _check_pair(a: Document, b: Document) -> Finding | None:
    a_vecs = _stack(a)
    b_vecs = _stack(b)
    if a_vecs.size == 0 or b_vecs.size == 0:
        return None

    sim = a_vecs @ b_vecs.T  # (len(a), len(b))
    a_max = sim.max(axis=1)  # best match in B for each paragraph in A
    b_max = sim.max(axis=0)  # best match in A for each paragraph in B

    a_hits = int((a_max >= NEAR_DUP_COSINE).sum())
    b_hits = int((b_max >= NEAR_DUP_COSINE).sum())

    a_ratio = a_hits / len(a.paragraphs)
    b_ratio = b_hits / len(b.paragraphs)
    max_ratio = max(a_ratio, b_ratio)

    if max_ratio < NEAR_DUP_PROPORTION:
        return None

    # Order so the higher-ratio document is reported as primary.
    primary, other = (a, b) if a_ratio >= b_ratio else (b, a)
    primary_ratio = max(a_ratio, b_ratio)
    other_ratio = min(a_ratio, b_ratio)

    # Enumerate matched paragraph pairs for evidence.
    matched_pairs: list[tuple[int, int, float]] = []
    p_vecs = _stack(primary)
    o_vecs = _stack(other)
    p_sim = p_vecs @ o_vecs.T
    best_other = p_sim.argmax(axis=1)
    for p_idx, o_idx in enumerate(best_other):
        s = float(p_sim[p_idx, o_idx])
        if s >= NEAR_DUP_COSINE:
            matched_pairs.append((p_idx, int(o_idx), s))

    detail = (
        f"{primary_ratio:.0%} of paragraphs in `{primary.rel_path}` ({primary.classification.value}) "
        f"are near-identical to paragraphs in `{other.rel_path}` ({other.classification.value}); "
        f"{other_ratio:.0%} in the opposite direction."
    )
    fix = (
        f"Remove the lower-aligned copy, or align classifications. If `{primary.rel_path}` "
        f"is the canonical public-facing version, the {other.classification.value} draft should be archived or redacted."
    )
    return Finding(
        rule_id="R002",
        rule_name="near-duplicate-across-classifications",
        severity=Severity.MEDIUM,
        primary_file=primary.rel_path,
        related_files=[other.rel_path],
        detail=detail,
        fix=fix,
        score=primary_ratio,
        evidence={
            "primary_path": primary.rel_path,
            "other_path": other.rel_path,
            "primary_classification": primary.classification.value,
            "other_classification": other.classification.value,
            "primary_ratio": primary_ratio,
            "other_ratio": other_ratio,
            "matched_pairs": matched_pairs,
        },
    )


def _stack(doc: Document) -> np.ndarray:
    if not doc.paragraphs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack([p.embedding for p in doc.paragraphs])
