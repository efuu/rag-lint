"""Core data models for rag-lint."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Classification(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


CLASSIFICATION_RANK = {
    Classification.PUBLIC: 0,
    Classification.INTERNAL: 1,
    Classification.CONFIDENTIAL: 2,
    Classification.RESTRICTED: 3,
}


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


SEVERITY_RANK = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}


class Paragraph(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: int
    text: str
    start_line: int
    embedding: Optional[np.ndarray] = None


class Document(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path
    rel_path: str
    classification: Optional[Classification] = None
    doc_type: Optional[str] = None
    date: Optional[str] = None
    raw_text: str
    body: str
    paragraphs: list[Paragraph] = Field(default_factory=list)


class Finding(BaseModel):
    rule_id: str
    rule_name: str
    severity: Severity
    primary_file: str
    related_files: list[str] = Field(default_factory=list)
    detail: str
    fix: str
    score: float = 0.0
    evidence: dict = Field(default_factory=dict)

    def sort_key(self) -> tuple[int, float, str]:
        return (SEVERITY_RANK[self.severity], -self.score, self.primary_file)
