# rag-lint

A linter for RAG corpus access controls. Three rules, run before indexing, stdout-first.

rag-lint reads a directory of Markdown documents, checks classification frontmatter, and flags pairs of documents whose paragraph-level overlap creates the precondition for a paraphrase leak across classification boundaries. It runs entirely offline. No LLM calls at runtime.

---

## What it checks

| Rule | Name | Severity | Flags |
|------|------|----------|-------|
| **R001** | `missing-classification` | high | Document has no `classification` field in frontmatter. |
| **R002** | `near-duplicate-across-classifications` | medium | Two documents at different classifications share ≥30% of paragraphs at cosine ≥0.90. |
| **R003** | `cross-classification-semantic-overlap` | high | Two documents at different classifications have a paragraph-pair cosine ≥0.75. Suppressed when R002 fires for the same pair. |

All three fire at index time on the Markdown corpus. They do not watch runtime retrieval.

## Threat model

In a RAG pipeline, two access-control failure modes survive typical per-document ACLs:

1. A document with no classification gets indexed under a default that is more permissive than intended.
2. A document at a higher classification shares enough paragraph-level content with a lower-classification document that a well-phrased query against the lower document can surface paraphrased restricted material without the retriever ever crossing the ACL boundary.

rag-lint catches the precondition for both. It does not prove a leak has occurred; it proves the corpus is shaped in a way that makes one possible.

## Install

Requires Python 3.11+. Uses [uv](https://docs.astral.sh/uv/).

```
git clone <this-repo> rag-lint
cd rag-lint
uv sync
```

First run downloads the MiniLM embedding model (≈90 MB) from Hugging Face. Subsequent runs are offline.

## Usage

```
# stdout (default)
uv run rag-lint path/to/corpus

# HTML with side-by-side R003 view
uv run rag-lint path/to/corpus --format html -o report.html
```

Exit codes: `0` clean, `1` findings present, `2` usage error.

### Corpus layout

Corpus is a directory of `*.md` files (recursive). Each document should carry YAML frontmatter:

```
---
classification: restricted     # public | internal | confidential | restricted
doc_type: ma_memo              # optional, display only
date: 2026-01-14               # optional, display only
---
```

Paragraphs are split on blank lines. Headers are ignored. No tagging or compartment model in v1.

## Example

A small financial-services corpus ships under `data/corpus/acme_financial/`:

```
uv run rag-lint data/corpus/acme_financial
```

Expect three findings: one R001 on an undeclared draft, one R002 on a near-duplicate policy pair, one R003 on an M&A memo whose strategic-rationale section reuses public annual-report phrasing.

Baseline fixtures from that run are committed under `runs/fixtures/` for quick inspection.

## Extending

Rules implement `rag_lint.rules.base.Rule` and produce `Finding` objects:

```python
class Rule(ABC):
    id: str
    name: str
    severity: Severity

    @abstractmethod
    def check(self, documents: list[Document], prior_findings: list[Finding]) -> list[Finding]: ...
```

`prior_findings` is the accumulated list across earlier rules in this run. Use it to implement precedence (R003 inspects it to suppress duplicates of R002). Wire a new rule into `cli.py`.

### Thresholds

Thresholds (`NEAR_DUP_COSINE = 0.90`, `NEAR_DUP_PROPORTION = 0.30`, `OVERLAP_COSINE = 0.75`) are module constants in the rule files. They were tuned once against the shipped corpus and locked. There is no config file in v1 by design — a linter with per-run threshold tuning has the wrong shape.

## Limitations

rag-lint is deliberately narrow. It does not check for:

- PII, credentials, or secret leakage
- OCR accuracy or missing content
- Document freshness or staleness
- Contradictions between documents
- Classification inference from content (the label is trusted; missing labels are flagged)
- Tag or compartment models beyond the four-level classification hierarchy
- Formats other than `.md` (no PDF, DOCX, or HTML ingestion)

It also does not watch runtime retrieval. Pipeline-side filters are still required; rag-lint reduces the surface area those filters have to get right.

## CI

rag-lint exits `1` on findings and prints a stable stdout format. A minimal gate:

```yaml
- run: uv run rag-lint data/corpus
```

The HTML report (`--format html`) is a single static file suitable for publishing as a build artifact.

## Notes

- No runtime LLM calls. Embedding is local via `sentence-transformers/all-MiniLM-L6-v2`.
- No caching layer. Reruns re-embed. The shipped corpus embeds in under two seconds after the model warm-up.
- No test suite; the corpus under `data/corpus/acme_financial/` is the test. Cold-boot dry run with `rm -rf .venv && uv sync && uv run rag-lint data/corpus/acme_financial` reproduces the baseline fixtures.
