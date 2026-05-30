"""
Build, save, and load a BM25Okapi sparse index over proposition texts.
Handles tokenization internally for pipeline compatibility.
"""

import pickle
import re
import logging
from pathlib import Path
from typing import List, Dict

__all__ = ["build_bm25_index", "save_bm25_index", "load_bm25_index", "bm25_retrieve", "tokenize_for_bm25"]

logger = logging.getLogger(__name__)


def tokenize_for_bm25(text: str) -> List[str]:
    """Lowercase, remove punctuation, split on whitespace."""
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return text.split()


def build_bm25_index(propositions: List[Dict], k1: float = 1.5, b: float = 0.75):
    """Build a BM25Okapi index from proposition texts.

    Args:
        propositions: List of proposition dicts with "text" key
        k1, b: BM25 hyperparameters

    Returns:
        rank_bm25.BM25Okapi instance.
    """
    from rank_bm25 import BM25Okapi
    
    # Tokenize all proposition texts internally
    tokenized_corpus = [tokenize_for_bm25(p["text"]) for p in propositions]
    
    index = BM25Okapi(tokenized_corpus, k1=k1, b=b)
    logger.info("BM25 index built on %d documents", len(tokenized_corpus))
    return index


def save_bm25_index(index, path: str | Path) -> None:
    """Pickle a BM25 index to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f)
    logger.info("BM25 index saved to %s", path)


def load_bm25_index(path: str | Path):
    """Load a pickled BM25 index from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"BM25 index not found: {path}")
    with open(path, "rb") as f:
        index = pickle.load(f)
    logger.info("BM25 index loaded from %s", path)
    return index


def bm25_retrieve(query: str, bm25_index, keyword_expansion: List[str] | None = None,
                  top_k: int = 20) -> List[tuple[int, float]]:
    """Query a BM25 index with optional keyword expansion.

    Args:
        query: Raw query string.
        bm25_index: BM25Okapi instance.
        keyword_expansion: Extra tokens to append to query.
        top_k: Number of results to return.

    Returns:
        List of (proposition_index, score) sorted descending, score > 0 only.
    """
    import numpy as np

    tokens = tokenize_for_bm25(query)
    if keyword_expansion:
        for kw in keyword_expansion:
            tokens += tokenize_for_bm25(kw)

    scores = bm25_index.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]


if __name__ == "__main__":
    import tempfile, os

    # Mock propositions
    mock_props = [
        {"text": "Vendor lock-in reduces procurement flexibility."},
        {"text": "Data privacy requires explicit consent under DPDP."},
        {"text": "AI governance frameworks emphasize transparency."},
    ]
    
    idx = build_bm25_index(mock_props)

    results = bm25_retrieve("vendor lock", idx, top_k=3)
    assert len(results) > 0
    assert results[0][0] == 0  # First doc should match "vendor lock"

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "bm25.pkl")
        save_bm25_index(idx, p)
        idx2 = load_bm25_index(p)
        r2 = bm25_retrieve("vendor", idx2, top_k=2)
        assert len(r2) > 0

    print("✅ bm25_builder tests passed")