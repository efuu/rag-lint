# rag-lint

A linter for RAG corpus access controls. Catches three classification-boundary bugs before you index a document folder.

Runs offline. No LLM calls at runtime.

## What it checks

| Rule | Severity | Flags |
|------|----------|-------|
| **R001** · `missing-classification` | high | Document has no `classification` in frontmatter. |
| **R002** · `near-duplicate-across-classifications` | medium | Two docs at different classifications are ≥30% near-identical paragraphs (cosine ≥0.90). |
| **R003** · `cross-classification-semantic-overlap` | high | Two docs at different classifications share a paragraph (cosine ≥0.75). Suppressed when R002 fires for the same pair. |

**R001 catches "never labeled." R002 catches "wrong label on the whole doc." R003 catches "wrong label on one paragraph."**

## Threat model

Per-document ACLs work per-document. Documents have relationships — missing labels, misclassifications, cross-classification paragraph overlap — that per-document ACLs don't see. rag-lint runs before indexing and catches the three corpus-shape conditions that make ACL-based safety fragile.

It catches the **precondition** for a paraphrase leak across classifications. It does not prove a leak has occurred.

## Install

Python 3.11+, [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/efuu/rag-lint
cd rag-lint
uv sync
```

First run downloads MiniLM (~90 MB). Subsequent runs are offline.

## Usage

```
# stdout (default)
uv run rag-lint path/to/corpus

# HTML report with side-by-side hero view
uv run rag-lint path/to/corpus --format html -o report.html
```

Exit codes: `0` clean, `1` findings present, `2` usage error.

Frontmatter per document:

```
---
classification: restricted   # public | internal | confidential | restricted
---
```

## Example

A 14-document financial-services corpus ships under `data/corpus/acme_financial/`:

```
uv run rag-lint data/corpus/acme_financial
```

Expect three findings (one per rule). Baselines committed in `runs/fixtures/`.

## Probe harness

`examples/probe_leak.py` takes the threat model out of prose and into numbers. Embeds the corpus, runs queries under three scenarios — correct ACL, failed ACL, failed ACL after applying R003's redaction prescription — and reports the change in retrieval rank and cosine.

```
uv run python examples/probe_leak.py
```

On the shipped corpus: top restricted cosine drops from **0.72 → 0.66** after R003 redaction on the overlap-targeting query, with the specific doorway paragraph falling out of top-5. That's the blast-radius reduction, quantified.
