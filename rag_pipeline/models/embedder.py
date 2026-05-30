# FILE: rag_pipeline/models/embedder.py
"""
Embedder wrapper for sentence-transformers with caching.
"""

import logging
from pathlib import Path
from typing import Optional, Union, List
import numpy as np

from sentence_transformers import SentenceTransformer

from ..config import EMBEDDING_MODEL, DEVICE

__all__ = ["EmbedderWrapper", "load_embedder"]

logger = logging.getLogger(__name__)


class EmbedderWrapper:
    """Wrapper around SentenceTransformer with caching and batch support."""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or EMBEDDING_MODEL
        self.device = device or DEVICE
        self._model: Optional[SentenceTransformer] = None
    
    def _load(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            logger.info(f"Loading embedder: {self.model_name} on {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        normalize: bool = True,
        batch_size: int = 32,
        **kwargs
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: Single string or list of strings
            normalize: L2-normalize embeddings (required for cosine/inner-product)
            batch_size: Batch size for encoding
            **kwargs: Passed to model.encode()
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        self._load()
        
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        # Encode with progress bar disabled by default
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
            **kwargs
        )
        
        return np.array(embeddings, dtype=np.float32)
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        self._load()
        return self._model.get_sentence_embedding_dimension()


def load_embedder(
    model_name: Optional[str] = None, 
    device: Optional[str] = None
) -> EmbedderWrapper:
    """Convenience function to create and return an EmbedderWrapper."""
    return EmbedderWrapper(model_name=model_name, device=device)


# ── Test ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    embedder = load_embedder()
    
    # Test single text
    emb = embedder.encode("Vendor lock-in reduces flexibility")
    assert emb.shape == (384,), f"Expected (384,), got {emb.shape}"
    
    # Test batch
    embs = embedder.encode(["Text 1", "Text 2", "Text 3"])
    assert embs.shape == (3, 384), f"Expected (3, 384), got {embs.shape}"
    
    # Test normalization (unit vectors)
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0), "Embeddings should be normalized"
    
    print(f"✅ embedder tests passed — dim={embedder.dimension}")