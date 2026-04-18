"""Paragraph embeddings using sentence-transformers MiniLM."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import warnings

import numpy as np

from rag_lint.models import Document

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def _silence_libraries() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    for name in (
        "transformers",
        "sentence_transformers",
        "huggingface_hub",
        "urllib3",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


@contextlib.contextmanager
def _silence_fds():
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(devnull)


def _get_model():
    global _model
    if _model is None:
        _silence_libraries()
        with _silence_fds():
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_paragraphs(documents: list[Document]) -> None:
    """Compute paragraph embeddings in place. L2-normalized for cosine via dot product."""
    flat_texts: list[str] = []
    flat_refs: list[tuple[int, int]] = []
    for d_idx, doc in enumerate(documents):
        for p_idx, para in enumerate(doc.paragraphs):
            flat_texts.append(para.text)
            flat_refs.append((d_idx, p_idx))

    if not flat_texts:
        return

    model = _get_model()
    vectors = model.encode(
        flat_texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    for (d_idx, p_idx), vec in zip(flat_refs, vectors):
        documents[d_idx].paragraphs[p_idx].embedding = np.asarray(vec, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
