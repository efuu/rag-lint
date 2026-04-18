# Examples

## `probe_leak.py` — empirical blast-radius check

A small probe harness that demonstrates what R003 actually buys you in the
presence of an ACL failure.

**Setup.** The shipped corpus is embedded. Queries a public-cleared user might
plausibly ask are run against the index under three scenarios:

- **A.** Correct ACL — restricted docs filtered out.
- **B.** ACL failed — all docs visible to the retriever.
- **C.** ACL failed, *but* we've applied the rag-lint R003 prescription
  (redact the overlapping paragraphs from the higher-classification docs).

The interesting contrast is B vs C: when the ACL fails, does reducing
cross-class paragraph overlap change the leak surface?

**Run:**

```
uv run python examples/probe_leak.py
```

**Baseline fixture:** `runs/fixtures/probe_leak.txt`.

### What it shows

On the shipped corpus:

- Queries targeting R003-overlapping content see the top restricted cosine
  drop by ~0.06 after redaction, with the specific "doorway" paragraph
  falling out of the top-5 and being replaced by a lower-scoring restricted
  paragraph. That's the blast-radius reduction.
- Queries that match restricted content *unrelated* to R003 findings (e.g.,
  a question about deal valuation that hits deal-specific sections which
  don't exist publicly) show no change. Correct behavior — R003 is a
  surgical tool, not a blanket suppressor.

### What this proves, and what it doesn't

**Proves:** R003's redaction prescription measurably reduces the rank and
score of the specific doorway content when the ACL fails.

**Does not prove:** that R003 alone would stop a motivated attacker — the
restricted docs still contain deal-specific content that matches queries
aimed at that content. R003 is defense-in-depth against ACL failure for the
*doorway* paragraphs, not a replacement for the ACL.

The probe is here so the linter's threat model is testable in the open
rather than asserted in prose.
