"""
Standalone RAPTOR retrieval helper used when intent is broad (general/comparative).
Wraps the FAISS search over RAPTOR summary nodes.
"""

import logging
import numpy as np

__all__ = ["RaptorRetriever"]

logger = logging.getLogger(__name__)


class RaptorRetriever:
    """Query the RAPTOR summary hierarchy for broad, cross-scenario context.

    Useful for 'general' or 'comparative' intent queries where individual
    proposition hits are insufficient.

    Args:
        raptor_faiss: FAISS index over RAPTOR summary embeddings.
        raptor_ids: Ordered list of summary IDs aligned to FAISS rows.
        raptor_summaries: dict[summary_id → {text, level, child_ids, scenario_id}].
        embedder: Embedder instance for query encoding.

    Example:
        >>> retriever = RaptorRetriever(faiss_idx, ids, summaries, embedder)
        >>> hits = retriever.search("AI governance overview", top_k=2)
        >>> isinstance(hits[0][0], str)
        True
    """

    def __init__(self, raptor_faiss, raptor_ids: list[str],
                 raptor_summaries: dict[str, dict], embedder):
        self.raptor_faiss     = raptor_faiss
        self.raptor_ids       = raptor_ids
        self.raptor_summaries = raptor_summaries
        self.embedder         = embedder

    def search(self, query: str, top_k: int = 2) -> list[tuple[str, float]]:
        """Retrieve top RAPTOR summary nodes for a broad query.

        Args:
            query: Query string.
            top_k: Number of summary nodes to return.

        Returns:
            List of (summary_id, score) tuples sorted by descending score.
        """
        q_emb    = self.embedder.encode([query], normalize=True)
        n_search = min(len(self.raptor_ids), max(top_k * 3, 10))

        try:
            scores, indices = self.raptor_faiss.search(q_emb, n_search)
        except Exception as exc:
            logger.error("RAPTOR search error: %s", exc)
            return []

        results: list[tuple[str, float]] = []
        for idx, sc in zip(indices[0], scores[0]):
            smid = self.raptor_ids[int(idx)]
            results.append((smid, float(sc)))

        return results[:top_k]


if __name__ == "__main__":
    class MockEmbedder:
        def encode(self, texts, normalize=True):
            vecs = np.ones((len(texts), 384), dtype="float32")
            if normalize:
                vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs

    class MockFAISS:
        def search(self, q, k):
            return np.ones((1, k), dtype="float32"), np.arange(k).reshape(1, k)

    summaries = {
        "raptor_L1_S1": {"text": "S1 summary", "level": 1, "scenario_id": "S1"},
        "raptor_L2_global": {"text": "global", "level": 2, "scenario_id": "global"},
    }
    ret = RaptorRetriever(MockFAISS(), list(summaries.keys()), summaries, MockEmbedder())
    hits = ret.search("governance overview", top_k=2)
    assert len(hits) == 2
    assert isinstance(hits[0][0], str)
    print("✅ raptor_retriever tests passed")