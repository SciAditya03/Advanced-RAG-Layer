# FILE: rag_pipeline/retrieval/hybrid_retriever.py
"""
Optimized tiered hybrid retriever with pre-loading and timeout protection.
"""

import logging
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from threading import Lock

from ..config import (
    RRF_K, BM25_K1, BM25_B, TOP_K_RAPTOR, 
    EMBEDDING_MODEL, DEVICE, PRELOAD_EMBEDDER, 
    EMBEDDER_LOAD_TIMEOUT, USE_ONNX_RUNTIME, BM25_ONLY_MODE
)
from ..utils.helpers import tokenize_for_bm25

__all__ = ["TieredHybridRetriever"]

logger = logging.getLogger(__name__)


class TieredHybridRetriever:
    """Optimized hybrid retriever with pre-loading and graceful fallbacks."""
    
    def __init__(
        self,
        faiss_index,
        bm25_index,
        raptor_faiss,
        raptor_metadata: Dict,
        parent_cache,
        rrf_k: int = None,
        bm25_k1: float = None,
        bm25_b: float = None,
        preload_embedder: bool = None
    ):
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.raptor_faiss = raptor_faiss
        self.raptor_metadata = raptor_metadata
        self.parent_cache = parent_cache
        
        self.rrf_k = rrf_k or RRF_K
        self.bm25_k1 = bm25_k1 or BM25_K1
        self.bm25_b = bm25_b or BM25_B
        
        # Thread-safe embedder loading
        self._query_embedder = None
        self._embedder_lock = Lock()
        self._embedder_loaded = False
        self._embedder_load_error = None
        
        # Pre-load if configured
        preload = preload_embedder if preload_embedder is not None else PRELOAD_EMBEDDER
        if preload:
            self._preload_embedder_sync()
        
        logger.info("✅ TieredHybridRetriever initialized")
    
    def _preload_embedder_sync(self):
        """Pre-load embedder synchronously at init (with timeout)."""
        start = time.time()
        try:
            with self._embedder_lock:
                if self._query_embedder is not None:
                    return
                logger.info(f"🔄 Pre-loading embedder: {EMBEDDING_MODEL}...")
                from sentence_transformers import SentenceTransformer
                self._query_embedder = SentenceTransformer(
                    EMBEDDING_MODEL, device=DEVICE, local_files_only=True
                )
                elapsed = time.time() - start
                self._embedder_loaded = True
                logger.info(f"✅ Embedder pre-loaded in {elapsed:.1f}s")
        except Exception as e:
            self._embedder_load_error = str(e)
            logger.warning(f"⚠️  Embedder pre-load failed: {e}")
            logger.warning("   Will fallback to BM25-only mode")
    
    def _get_embedder(self):
        """Get embedder instance with lazy-loading + timeout protection."""
        if self._embedder_loaded:
            return self._query_embedder
        
        with self._embedder_lock:
            if self._query_embedder is not None:
                return self._query_embedder
            if self._embedder_load_error:
                return None
            
            import signal
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Embedder load exceeded {EMBEDDER_LOAD_TIMEOUT}s")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(EMBEDDER_LOAD_TIMEOUT)
            
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"🔄 Loading embedder: {EMBEDDING_MODEL}...")
                self._query_embedder = SentenceTransformer(
                    EMBEDDING_MODEL, device=DEVICE, local_files_only=True
                )
                self._embedder_loaded = True
                logger.info("✅ Embedder loaded")
                return self._query_embedder
            except TimeoutError:
                logger.error(f"❌ Embedder load timed out after {EMBEDDER_LOAD_TIMEOUT}s")
                self._embedder_load_error = "timeout"
                return None
            except Exception as e:
                logger.error(f"❌ Embedder load failed: {e}")
                self._embedder_load_error = str(e)
                return None
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
    
    def _encode_query(self, query_text: str) -> Optional[np.ndarray]:
        """Encode query with embedder, with fallback handling."""
        embedder = self._get_embedder()
        if embedder is None:
            return None
        try:
            emb = embedder.encode(
                [query_text], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
            )
            return emb.astype("float32")
        except Exception as e:
            logger.error(f"❌ Query encoding failed: {e}")
            return None
    
    def _dense_search(self, query_text: str, valid_ids: Set[int], top_k: int = 50) -> List[Tuple[int, float]]:
        """FAISS dense search with timeout protection."""
        if BM25_ONLY_MODE:
            logger.debug("⚠️  Dense search skipped (BM25_ONLY_MODE=True)")
            return []
        
        start = time.time()
        q_emb = self._encode_query(query_text)
        if q_emb is None:
            return []
        
        try:
            n_search = min(self.faiss_index.ntotal, max(top_k * 5, 200))
            scores, indices = self.faiss_index.search(q_emb, n_search)
            results = []
            for idx, score in zip(indices[0], scores[0]):
                prop_idx = int(idx)
                if prop_idx in valid_ids:
                    results.append((prop_idx, float(score)))
                if len(results) >= top_k:
                    break
            elapsed = time.time() - start
            logger.debug(f"🔍 Dense search: {len(results)} results in {elapsed*1000:.0f}ms")
            return results
        except Exception as e:
            logger.error(f"❌ Dense search failed: {e}")
            return []
    
    def _sparse_search(self, query_text: str, extra_kws: List[str], valid_ids: Set[int], top_k: int = 50) -> List[Tuple[int, float]]:
        """BM25 sparse search — fast and reliable."""
        start = time.time()
        try:
            tokens = tokenize_for_bm25(query_text)
            if extra_kws:
                for kw in extra_kws:
                    tokens += tokenize_for_bm25(kw)
            all_scores = self.bm25_index.get_scores(tokens)
            results = []
            for prop_idx in valid_ids:
                if prop_idx < len(all_scores):
                    score = float(all_scores[prop_idx])
                    if score > 0:
                        results.append((prop_idx, score))
            results.sort(key=lambda x: -x[1])
            results = results[:top_k]
            elapsed = time.time() - start
            logger.debug(f"🔍 Sparse search: {len(results)} results in {elapsed*1000:.0f}ms")
            return results
        except Exception as e:
            logger.error(f"❌ Sparse search failed: {e}")
            return []
    
    def _raptor_search(self, query_text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """RAPTOR summary search — optional, skip if slow."""
        if BM25_ONLY_MODE:
            return []
        if not self.raptor_metadata or self.raptor_faiss.ntotal == 0:
            return []
        q_emb = self._encode_query(query_text)
        if q_emb is None:
            return []
        try:
            n_search = min(self.raptor_faiss.ntotal, top_k * 3)
            scores, indices = self.raptor_faiss.search(q_emb, n_search)
            summary_ids = list(self.raptor_metadata.keys())
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if int(idx) < len(summary_ids):
                    smid = summary_ids[int(idx)]
                    results.append((smid, float(score)))
            return results[:top_k]
        except Exception as e:
            logger.debug(f"⚠️  RAPTOR search skipped: {e}")
            return []
    
    @staticmethod
    def _reciprocal_rank_fusion(ranked_lists: List[List[Tuple[int, float]]], k: int = None) -> Dict[int, float]:
        """RRF fusion with empty-list handling."""
        k = k or RRF_K
        fused = {}
        for ranked in ranked_lists:
            for rank, (doc_id, _) in enumerate(ranked):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return fused
    
    def search(self, query: str, officer_profile: Dict, extra_kws: Optional[List[str]] = None,
               intent: str = "specific", top_k: int = 10) -> Tuple[List[Dict], List[Tuple[str, float]]]:
        """Main search method with performance logging."""
        start = time.time()
        extra_kws = extra_kws or []
        filters = {k: officer_profile[k] for k in ["level", "scenario_id"] if k in officer_profile}
        total_props = len(self.parent_cache) * 11
        valid_ids = self._apply_metadata_filters(filters, total_props)
        
        dense_results = self._dense_search(query, valid_ids, top_k=50)
        sparse_results = self._sparse_search(query, extra_kws, valid_ids, top_k=50)
        
        ranked_lists = [r for r in [dense_results, sparse_results] if r]
        rrf_scores = self._reciprocal_rank_fusion(ranked_lists, k=self.rrf_k) if ranked_lists else {}
        sorted_results = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]
        
        hits = []
        chunk_keys = list(self.parent_cache.chunks.keys())
        for prop_idx, rrf_score in sorted_results:
            chunk_idx = min(prop_idx // 11, len(chunk_keys) - 1)
            cid = chunk_keys[chunk_idx]
            chunk = self.parent_cache.chunks.get(cid, {})
            hits.append({
                "prop_idx": prop_idx, "parent_id": cid,
                "scenario_id": chunk.get("scenario_id", "unknown"),
                "level": chunk.get("difficulty_level", "all"),
                "topic": chunk.get("topic", "unknown"),
                "rrf_score": rrf_score,
                "snippet": chunk.get("detailed_explanation", "")[:100]
            })
        
        raptor_hits = []
        if intent in ("general", "comparative") and not BM25_ONLY_MODE:
            raptor_hits = self._raptor_search(query, top_k=TOP_K_RAPTOR)
        
        elapsed = time.time() - start
        logger.info(f"🔍 Retrieval complete: {len(hits)} hits in {elapsed*1000:.0f}ms")
        return hits, raptor_hits
    
    def _apply_metadata_filters(self, filters: Dict[str, str], total_props: int) -> Set[int]:
        """Filter proposition indices by metadata."""
        if not filters:
            return set(range(total_props))
        valid = set()
        for idx, (cid, chunk) in enumerate(self.parent_cache.chunks.items()):
            if filters.get("level"):
                chunk_level = chunk.get("difficulty_level", "all")
                if chunk_level not in ("all", filters["level"]):
                    continue
            if filters.get("scenario_id") and chunk.get("scenario_id") != filters["scenario_id"]:
                continue
            for offset in range(11):
                prop_idx = idx * 11 + offset
                if prop_idx < total_props:
                    valid.add(prop_idx)
        return valid if valid else set(range(total_props))


# ── Test ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    class MockIndex:
        def __init__(self, n): self.ntotal = n
        def search(self, q_emb, k):
            import numpy as np
            return (np.array([[0.9 - i*0.1 for i in range(k)]], dtype=np.float32),
                    np.array([[i for i in range(k)]], dtype=np.int64))
    class MockBM25:
        def get_scores(self, tokens):
            import numpy as np
            return np.array([0.5, 0.3, 0.8, 0.1, 0.6])
    class MockCache:
        def __init__(self):
            self.chunks = {f"chunk_{i:04d}": {"scenario_id": "IN-AIGOV-001", "topic": "test",
                "difficulty_level": "all", "detailed_explanation": "Test."} for i in range(5)}
        def get(self, k, d=None): return self.chunks.get(k, d)
        def items(self): return self.chunks.items()
        def __len__(self): return len(self.chunks)
    
    retriever = TieredHybridRetriever(
        faiss_index=MockIndex(50), bm25_index=MockBM25(), raptor_faiss=MockIndex(5),
        raptor_metadata={"raptor_L1_test": {"text": "summary", "level": 1}},
        parent_cache=MockCache(), preload_embedder=False
    )
    hits, raptor = retriever.search("test", {"level": "mid", "scenario_id": "IN-AIGOV-001"}, top_k=3)
    assert len(hits) <= 3
    print(f"✅ hybrid_retriever tests passed — {len(hits)} hits")