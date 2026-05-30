# FILE: rag_pipeline/pipeline/governance_rag.py
"""
Main GovernanceRAGPipeline class: end-to-end RAG execution.
Loads indices from disk, handles query preprocessing, retrieval, 
context compression, and answer generation.
"""

import logging, time, re
from pathlib import Path
from typing import List, Dict, Optional

from ..config import (
    FAISS_INDEX_PATH, BM25_INDEX_PATH, RAPTOR_FAISS_PATH, RAPTOR_METADATA_PATH,
    PARENT_CACHE_PATH, CONTEXT_BUDGET_CHARS, LEVEL_FIELD_MAP, STYLE_GUIDE, TOP_K_RAPTOR,
    PRELOAD_EMBEDDER, PRELOAD_LLM
)
from ..indexing.parent_cache import ParentCache
from ..indexing.faiss_builder import load_faiss_index
from ..indexing.bm25_builder import load_bm25_index
from ..indexing.raptor_builder import load_raptor_data
from ..retrieval.hybrid_retriever import TieredHybridRetriever
from ..retrieval.query_preprocessor import preprocess_query
from ..retrieval.query_rewriter import rewrite_query_with_history
from ..context.compressor import compress_for_officer
from ..context.prompt_builder import build_teaching_prompt
from ..models.llm_wrapper import generate_answer
from ..models.crag_verifier import verify_answer
from ..utils.output_formatter import format_for_teaching_section

__all__ = ["GovernanceRAGPipeline"]
logger = logging.getLogger(__name__)


class GovernanceRAGPipeline:
    """Full tiered RAG pipeline for AI Governance Simulator."""

    def __init__(self, indices_dir: Optional[Path] = None):
        """Initialize pipeline by loading all indices from disk."""
        t0 = time.time()
        logger.info("🔍 Initializing GovernanceRAGPipeline...")
        
        indices_dir = Path(indices_dir) if indices_dir else FAISS_INDEX_PATH.parent
        
        logger.info("📦 Loading parent cache...")
        self.parent_cache = ParentCache.load(PARENT_CACHE_PATH)
        
        logger.info("🧠 Loading FAISS index...")
        self.faiss_index = load_faiss_index(FAISS_INDEX_PATH)
        
        logger.info("📚 Loading BM25 index...")
        self.bm25_index = load_bm25_index(BM25_INDEX_PATH)
        
        logger.info("🌲 Loading RAPTOR indices...")
        self.raptor_faiss, self.raptor_metadata = load_raptor_data(indices_dir)
        
        logger.info("🔗 Initializing TieredHybridRetriever...")
        self.retriever = TieredHybridRetriever(
            faiss_index=self.faiss_index, bm25_index=self.bm25_index,
            raptor_faiss=self.raptor_faiss, raptor_metadata=self.raptor_metadata,
            parent_cache=self.parent_cache, preload_embedder=False
        )
        
        # Pre-load embedder if configured (for faster first query)
        if PRELOAD_EMBEDDER:
            try:
                self.retriever._preload_embedder_sync()
            except Exception as e:
                logger.warning(f"⚠️  Embedder pre-load failed: {e}")
        
        elapsed = time.time() - t0
        logger.info(f"✅ Pipeline initialized in {elapsed:.2f}s")
        logger.info(f"   • Parent chunks: {len(self.parent_cache)}")
        logger.info(f"   • FAISS vectors: {self.faiss_index.ntotal}")
        logger.info(f"   • RAPTOR nodes: {len(self.raptor_metadata)}")

    def run(self, query: str, officer_profile: dict, chat_history: Optional[List[dict]] = None,
            scenario_id: Optional[str] = None, max_retries: int = 1) -> dict:
        """Execute full RAG pipeline for a query."""
        t0 = time.time()
        chat_history = chat_history or []
        
        # 1. Query Rewriting
        rewritten_q = rewrite_query_with_history(query, chat_history, scenario_id)
        logger.info(f"🔄 Rewrote: '{query[:50]}...' → '{rewritten_q[:80]}...'")
        
        # 2. Preprocess query
        if scenario_id:
            officer_profile = {**officer_profile, "scenario_id": scenario_id}
        pq = preprocess_query(rewritten_q, officer_profile)
        logger.info(f"🔍 Intent: {pq['intent']} | Sub-queries: {len(pq['sub_queries'])}")
        
        # 3. Retrieve propositions
        all_hits, all_raptor_hits = [], []
        for sub_q in pq["sub_queries"]:
            hits, raptor_hits = self.retriever.search(
                query=sub_q, officer_profile=officer_profile,
                extra_kws=pq["keywords"], intent=pq["intent"], top_k=10,
            )
            all_hits.extend(hits)
            all_raptor_hits.extend(raptor_hits)
        
        # Deduplicate by parent_id
        seen_pids = {}
        for hit in all_hits:
            pid = hit["parent_id"]
            if pid not in seen_pids or hit["rrf_score"] > seen_pids[pid]["rrf_score"]:
                seen_pids[pid] = hit
        deduped_hits = sorted(seen_pids.values(), key=lambda x: -x["rrf_score"])[:10]
        logger.info(f"📄 Retrieved {len(deduped_hits)} unique parent chunks")
        
        # 4. Compress context
        context, citations = compress_for_officer(
            deduped_hits, all_raptor_hits, officer_profile, self.parent_cache,
            budget_chars=CONTEXT_BUDGET_CHARS
        )
        
        # 5. Build teaching prompt & generate answer
        prompt = build_teaching_prompt(context, query, officer_profile)
        answer = generate_answer(prompt)
        
        # 6. CRAG verification
        verification_log = verify_answer(context, answer, query)
        retry_count = 0
        if not verification_log.get("supported", True) and max_retries > 0:
            missing = verification_log.get("missing_context", [])
            if missing:
                retry_query = rewritten_q + " " + " ".join(missing[:2])
                logger.info(f"🔄 CRAG retry: {retry_query[:80]}...")
                retry_hits, retry_raptor = self.retriever.search(
                    retry_query, officer_profile, intent=pq["intent"], top_k=10
                )
                retry_context, retry_citations = compress_for_officer(
                    retry_hits, retry_raptor, officer_profile, self.parent_cache
                )
                merged_context = context + "\n\n[RETRY CONTEXT]\n" + retry_context
                answer = generate_answer(build_teaching_prompt(merged_context, query, officer_profile))
                citations += retry_citations
                verification_log = verify_answer(merged_context, answer, query)
                verification_log["retry_attempted"] = True
                retry_count = 1
        
        # 7. Extract reflection question
        refl_match = re.search(r'(Reflect(?:ion)?\s*[Qq]uestion.*?(?:\?|$))', answer, re.DOTALL)
        reflection_question = refl_match.group(1).strip() if refl_match else ""
        
        elapsed = time.time() - t0
        logger.info(f"✅ Done in {elapsed:.1f}s | Retries: {retry_count}")
        
        return {
            "answer": answer, "citations": citations, "verification_log": verification_log,
            "reflection_question": reflection_question, "retrieved_propositions": deduped_hits,
            "compressed_context": context, "rewritten_query": rewritten_q,
            "elapsed_s": round(elapsed, 2),
        }


# ── Demo entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🧪 GovernanceRAGPipeline Demo")
    print("=" * 60)
    pipe = GovernanceRAGPipeline()
    print("\n🔍 Test: Basic beginner query")
    result = pipe.run(
        query="What is vendor lock-in?",
        officer_profile={"level": "beginner", "domain": "procurement"},
        scenario_id="IN-AIGOV-001"
    )
    print(f"✅ Answer ({len(result['answer'])} chars): {result['answer'][:200]}...")
    print("\n🎉 Demo completed!")