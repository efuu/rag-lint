"""Rule base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_lint.models import Document, Finding, Severity


class Rule(ABC):
    id: str
    name: str
    severity: Severity

    @abstractmethod
    def check(
        self, documents: list[Document], prior_findings: list[Finding]
    ) -> list[Finding]: ...
