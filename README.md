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

## What rag-lint is not

- Not PII detection (use Presidio)
- Not runtime retrieval filtering (pipeline-side concern)
- Not classification inference (labels are trusted; missing labels are flagged)
- Not a platform — three rules, locked thresholds, no config file, no test directory, no JSON output, Markdown only

Cut list is longer than the feature list on purpose. Narrowness is the genre.

## Limits

Naive O(N²) document pairs. Fine to ~1k documents. Past that, swap in an ANN index.

Thresholds (0.30, 0.75, 0.90) are module constants tuned against the shipped corpus and locked. A linter with per-run threshold tuning has the wrong shape.

Catches lazy paraphrase reuse, not adversarial paraphrase. Trusts frontmatter labels at face value.
