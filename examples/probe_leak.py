"""
Probe harness — empirically demonstrates why rag-lint R003 matters.

Scenario: a public-clearance user querying a RAG system built over the shipped
corpus. We show:

    1. What the user sees under a correctly-configured ACL (restricted docs
       filtered). Baseline.

    2. What the user *would* see if the ACL failed and no filtering applied.
       This is the leak surface.

    3. How that leak surface changes after applying rag-lint R003's
       prescription (redact the overlapping paragraphs from restricted docs).

If R003 has any real value beyond its narrow threat model, the rank of
restricted content in the "no ACL" scenario should drop materially after
redaction — i.e., even when the ACL fails, cross-class retrieval is less
likely to surface restricted text.

Run: uv run python examples/probe_leak.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_lint.embeddings import embed_paragraphs
from rag_lint.loader import load_corpus
from rag_lint.rules.r001_missing_classification import R001
from rag_lint.rules.r002_near_duplicate import R002
from rag_lint.rules.r003_cross_class_overlap import R003

CORPUS = Path(__file__).resolve().parent.parent / "data/corpus/acme_financial"
TOP_K = 5
R003_REDACT_COSINE = 0.75

QUERIES = [
    # Queries a public-cleared user might plausibly ask. Some overlap with
    # restricted content, some don't.
    "What is our strategy for private credit?",
    "How will we accelerate the disciplined expansion into private credit?",
    "Are there acquisitions being considered to scale the private credit platform?",
    "What is the valuation on any pending strategic transaction?",
    "How does our code of conduct handle conflicts of interest?",
]


def build_index(documents):
    paragraphs = []
    vectors = []
    for doc in documents:
        for para in doc.paragraphs:
            paragraphs.append((doc, para))
            vectors.append(para.embedding)
    if not vectors:
        return paragraphs, np.zeros((0, 384), dtype=np.float32)
    return paragraphs, np.vstack(vectors)


def retrieve(query_vec, paragraphs, vectors, k=TOP_K, acl_public_only=False):
    if vectors.size == 0:
        return []
    sims = vectors @ query_vec
    order = np.argsort(-sims)
    results = []
    for i in order:
        doc, para = paragraphs[i]
        if acl_public_only:
            if doc.classification is None:
                continue
            if doc.classification.value != "public":
                continue
        results.append((doc, para, float(sims[i])))
        if len(results) >= k:
            break
    return results


def format_result(doc, para, sim, mark_blocked_by_acl=False):
    cls = doc.classification.value if doc.classification else "unlabeled"
    marker = ""
    if mark_blocked_by_acl and cls != "public":
        marker = f"  ← {cls.upper()}, would be blocked by ACL"
    elif cls != "public":
        marker = f"  [{cls}]"
    return f"  {doc.rel_path}:{para.start_line:>3}   cosine {sim:.2f}{marker}"


def print_query_block(
    query, paragraphs, vectors, paragraphs_redacted, vectors_redacted, model
):
    q_vec = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)

    print()
    print(f"Query: {query!r}")
    print("-" * 78)

    with_acl = retrieve(q_vec, paragraphs, vectors, acl_public_only=True)
    print("  A. Top-5 retrieved under correct ACL (public user):")
    for doc, para, sim in with_acl:
        print(format_result(doc, para, sim))
    if not with_acl:
        print("  (no results)")

    no_acl = retrieve(q_vec, paragraphs, vectors, acl_public_only=False)
    print("\n  B. Top-5 retrieved if ACL failed (no filtering):")
    for doc, para, sim in no_acl:
        print(format_result(doc, para, sim, mark_blocked_by_acl=True))

    no_acl_redacted = retrieve(
        q_vec, paragraphs_redacted, vectors_redacted, acl_public_only=False
    )
    print("\n  C. Top-5 after applying rag-lint R003 redactions, still no ACL:")
    for doc, para, sim in no_acl_redacted:
        print(format_result(doc, para, sim, mark_blocked_by_acl=True))

    def top_restricted(results):
        for r, (doc, _, sim) in enumerate(results):
            cls = doc.classification.value if doc.classification else "unlabeled"
            if cls != "public":
                return r + 1, sim, doc.rel_path
        return None, None, None

    br, bs, bp = top_restricted(no_acl)
    ar, as_, ap = top_restricted(no_acl_redacted)
    print()
    if br and ar:
        print(
            f"  Blast-radius delta: top non-public hit "
            f"rank #{br} cos {bs:.2f} ({bp})  →  "
            f"rank #{ar} cos {as_:.2f} ({ap})"
        )
    elif br and not ar:
        print(
            f"  Blast-radius delta: top non-public hit "
            f"rank #{br} cos {bs:.2f} ({bp})  →  dropped out of top-{TOP_K}"
        )
    else:
        print("  Blast-radius delta: no non-public hits before or after")


def main() -> int:
    print("=" * 78)
    print("  rag-lint probe — does cross-class overlap actually change the")
    print("  leak surface when the ACL fails? Empirical check on the corpus.")
    print("=" * 78)

    documents = load_corpus(CORPUS)
    embed_paragraphs(documents)

    findings = []
    for rule_cls in (R001, R002, R003):
        findings.extend(rule_cls().check(documents, findings))

    redactions: set[tuple[str, int]] = set()
    for f in findings:
        if f.rule_id == "R003":
            for a_idx, b_idx, cos in f.evidence["all_pairs"]:
                if cos < R003_REDACT_COSINE:
                    continue
                if f.evidence["a_path"] == f.evidence["higher_path"]:
                    redactions.add((f.evidence["a_path"], a_idx))
                else:
                    redactions.add((f.evidence["b_path"], b_idx))

    print(f"\nCorpus: {len(documents)} documents")
    print(
        f"rag-lint findings: "
        f"{sum(1 for f in findings if f.rule_id == 'R001')} R001, "
        f"{sum(1 for f in findings if f.rule_id == 'R002')} R002, "
        f"{sum(1 for f in findings if f.rule_id == 'R003')} R003"
    )
    print(f"R003 redactions (simulated): {len(redactions)} paragraph(s)")
    for path, idx in sorted(redactions):
        print(f"  - {path}  paragraph #{idx}")

    paragraphs, vectors = build_index(documents)
    paragraphs_redacted = [
        (doc, para)
        for doc, para in paragraphs
        if (doc.rel_path, para.index) not in redactions
    ]
    vectors_redacted = (
        np.vstack([p.embedding for _, p in paragraphs_redacted])
        if paragraphs_redacted
        else np.zeros((0, 384), dtype=np.float32)
    )

    print(
        f"\nIndex: {len(paragraphs)} paragraph embeddings "
        f"({len(paragraphs_redacted)} after R003 redaction)"
    )

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    for q in QUERIES:
        print_query_block(
            q, paragraphs, vectors, paragraphs_redacted, vectors_redacted, model
        )

    print()
    print("=" * 78)
    print(
        "  Reading: when (C) beats (B) — restricted content drops in rank or "
        "disappears\n  from top-5 — that's the blast-radius reduction R003 is "
        "actually buying you\n  in the presence of an ACL failure."
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
