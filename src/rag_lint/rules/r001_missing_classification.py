"""R001 — missing-classification."""

from __future__ import annotations

from rag_lint.models import Document, Finding, Severity
from rag_lint.rules.base import Rule


class R001(Rule):
    id = "R001"
    name = "missing-classification"
    severity = Severity.HIGH

    def check(
        self, documents: list[Document], prior_findings: list[Finding]
    ) -> list[Finding]:
        findings: list[Finding] = []
        for doc in documents:
            if doc.classification is None:
                findings.append(
                    Finding(
                        rule_id=self.id,
                        rule_name=self.name,
                        severity=self.severity,
                        primary_file=doc.rel_path,
                        detail="Frontmatter has no `classification` field — access controls cannot be enforced on an undeclared document.",
                        fix="Add `classification:` to frontmatter (public | internal | confidential | restricted).",
                        score=0.0,
                    )
                )
        return findings
