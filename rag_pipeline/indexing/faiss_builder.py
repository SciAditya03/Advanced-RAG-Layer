"""
Build, save, and load FAISS IndexFlatIP (cosine on normalised vectors).
Handles embedding generation internally for pipeline compatibility.
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Dict

__all__ = ["build_faiss_index", "save_faiss_index", "load_faiss_index"]

logger = logging.getLogger(__name__)


def build_faiss_index(
    propositions: List[Dict], 
    embedder, 
    device: str,
    batch_size: int = 64
):
    """Create a FAISS IndexFlatIP from proposition texts.

    Args:
        propositions: List of proposition dicts with "text" key
        embedder: SentenceTransformer instance with .encode() method
        device: "cuda" or "cpu"
        batch_size: Embedding batch size

    Returns:
        faiss.IndexFlatIP with all vectors added.
    """
    import faiss

    # Extract texts and generate embeddings
    prop_texts = [p["text"] for p in propositions]
    
    all_embeddings = []
    for i in range(0, len(prop_texts), batch_size):
        batch = prop_texts[i:i + batch_size]
        embs = embedder.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        all_embeddings.append(embs)
    
    embeddings = np.vstack(all_embeddings).astype("float32")
    
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(f"embeddings must be 2-D non-empty; got shape {embeddings.shape}")
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    logger.info("FAISS index built: %d vectors, dim=%d", index.ntotal, dim)
    return index


def save_faiss_index(index, path: str | Path) -> None:
    """Persist a FAISS index to disk."""
    import faiss
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    logger.info("FAISS index saved to %s", path)


def load_faiss_index(path: str | Path):
    """Load a FAISS index from disk."""
    import faiss
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found: {path}")
    index = faiss.read_index(str(path))
    logger.info("FAISS index loaded from %s (%d vectors)", path, index.ntotal)
    return index


if __name__ == "__main__":
    import tempfile, os
    from sentence_transformers import SentenceTransformer
    
    # Mock propositions
    mock_props = [
        {"text": "Vendor lock-in reduces flexibility in procurement."},
        {"text": "Data protection requires explicit consent under DPDP Act."},
    ]
    
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    idx = build_faiss_index(mock_props, embedder, "cpu")
    assert idx.ntotal == 2
    assert idx.d == 384  # all-MiniLM-L6-v2 dimension

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "test.faiss")
        save_faiss_index(idx, p)
        idx2 = load_faiss_index(p)
        assert idx2.ntotal == 2

    print("✅ faiss_builder tests passed")