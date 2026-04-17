"""rag-lint CLI entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="rag-lint",
    help="A linter for RAG corpus access controls. Catches classification-boundary bugs before indexing.",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def lint(
    corpus: Path = typer.Argument(..., help="Path to corpus directory of .md files."),
    fmt: str = typer.Option(
        "stdout", "--format", "-f", help="Output format: stdout | html."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output path for html format."
    ),
) -> None:
    """Lint a corpus directory for classification-boundary bugs."""
    from rag_lint.embeddings import embed_paragraphs
    from rag_lint.loader import load_corpus
    from rag_lint.reporters.html import render_html
    from rag_lint.reporters.stdout import render_stdout
    from rag_lint.rules.r001_missing_classification import R001
    from rag_lint.rules.r002_near_duplicate import R002
    from rag_lint.rules.r003_cross_class_overlap import R003

    if not corpus.exists() or not corpus.is_dir():
        typer.echo(f"error: corpus path is not a directory: {corpus}", err=True)
        raise typer.Exit(code=2)

    documents = load_corpus(corpus)
    if not documents:
        typer.echo(f"error: no .md documents found in {corpus}", err=True)
        raise typer.Exit(code=2)

    embed_paragraphs(documents)

    findings: list = []
    for rule_cls in (R001, R002, R003):
        findings.extend(rule_cls().check(documents, findings))

    findings.sort(key=lambda f: f.sort_key())

    if fmt == "stdout":
        exit_code = render_stdout(documents, findings, sys.stdout)
        raise typer.Exit(code=exit_code)
    if fmt == "html":
        if output is None:
            typer.echo("error: --output is required for html format", err=True)
            raise typer.Exit(code=2)
        render_html(documents, findings, output, corpus_path=corpus)
        typer.echo(f"wrote {output}")
        raise typer.Exit(code=0 if not findings else 1)
    typer.echo(f"error: unknown format: {fmt}", err=True)
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
