# FILE: rag_pipeline/indexing/parent_cache.py
"""
Registry for parent chunks parsed from RAG_DataSet.json.
Produces the canonical parent_chunks dict consumed by all other modules.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..utils.helpers import safe_get, _clean
from ..config import SCENARIO_SKIP_KEYS

__all__ = ["ParentCache", "build_parent_cache"]

logger = logging.getLogger(__name__)


class ParentCache:
    """Registry that holds the full metadata for every parent chunk.

    Usage::

        cache = ParentCache()
        cache.load_from_dataset(raw_data)
        # cache.chunks is dict[chunk_id → metadata_dict]

    Args:
        None

    Attributes:
        chunks: dict mapping chunk_id → metadata dict.
    """

    def __init__(self):
        self.chunks: Dict[str, Dict[str, Any]] = {}
        self._counter: int = 0

    def _register(self, scenario_id: str, topic: str, raw_chunk: dict) -> str:
        """Create and store one parent chunk entry.

        Args:
            scenario_id: Scenario key from the dataset.
            topic: Topic string for the chunk.
            raw_chunk: Raw dict from the dataset.

        Returns:
            Assigned chunk_id string.
        """
        cid = f"chunk_{self._counter:04d}"
        self._counter += 1

        self.chunks[cid] = {
            "chunk_id":                cid,
            "scenario_id":             scenario_id,
            "topic":                   topic,
            "beginner_explanation":    safe_get(raw_chunk, "beginner_explanation"),
            "operational_explanation": safe_get(raw_chunk, "operational_explanation"),
            "strategic_explanation":   safe_get(raw_chunk, "strategic_explanation"),
            "detailed_explanation":    safe_get(raw_chunk, "detailed_explanation"),
            "government_example":      safe_get(raw_chunk, "government_example"),
            "government_relevance":    safe_get(raw_chunk, "government_relevance"),
            "why_it_matters":          safe_get(raw_chunk, "why_it_matters"),
            "possible_officer_confusions": safe_get(raw_chunk, "possible_officer_confusions"),
            "reflection_questions":    safe_get(raw_chunk, "reflection_questions"),
            "relevant_laws":           safe_get(raw_chunk, "relevant_laws"),
            "governance_principles":   safe_get(raw_chunk, "governance_principles"),
            "risk_types":              safe_get(raw_chunk, "risk_types"),
            "retrieval_keywords":      safe_get(raw_chunk, "retrieval_keywords"),
            "retrieval_tags":          safe_get(raw_chunk, "retrieval_tags"),
            "difficulty_level":        safe_get(raw_chunk, "difficulty_level", default="all"),
        }
        return cid

    def load_from_dataset(self, data: dict) -> None:
        """Populate cache from a parsed RAG_DataSet.json dict.

        Processes both scenario knowledge_chunks and top-level beginner_terms.

        Args:
            data: Full parsed JSON dict from RAG_DataSet.json.
        """
        scenario_ids = [k for k in data.keys() if k not in SCENARIO_SKIP_KEYS]

        # Scenario knowledge chunks
        for sid in scenario_ids:
            scenario_obj = data[sid]
            for raw_chunk in scenario_obj.get("knowledge_chunks", []):
                topic = safe_get(raw_chunk, "topic", default=sid)
                self._register(sid, topic, raw_chunk)

        # Beginner terms
        for term_obj in data.get("beginner_terms", []):
            synthetic = {
                "beginner_explanation":    safe_get(term_obj, "simple_explanation"),
                "operational_explanation": safe_get(term_obj, "simple_explanation"),
                "strategic_explanation":   safe_get(term_obj, "simple_explanation"),
                "detailed_explanation":    safe_get(term_obj, "simple_explanation"),
                "government_example":      safe_get(term_obj, "government_example"),
                "why_it_matters":          safe_get(term_obj, "why_it_matters"),
                "difficulty_level":        safe_get(term_obj, "difficulty_level", default="beginner"),
                "retrieval_keywords":      safe_get(term_obj, "retrieval_keywords"),
                "retrieval_tags":          safe_get(term_obj, "retrieval_tags"),
                "reflection_questions":    "",
                "relevant_laws":           "",
                "governance_principles":   "",
            }
            self._register("beginner_terms", safe_get(term_obj, "term", default="term"), synthetic)

        logger.info("ParentCache loaded: %d chunks", len(self.chunks))

    def get(self, chunk_id: str, default: Optional[dict] = None) -> Optional[dict]:
        """Retrieve a chunk by ID.

        Args:
            chunk_id: e.g. 'chunk_0042'
            default: returned when key missing.

        Returns:
            Chunk metadata dict or default.
        """
        return self.chunks.get(chunk_id, default)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, key: str) -> dict:
        return self.chunks[key]

    def __iter__(self):
        return iter(self.chunks.items())

    # ── Persistence methods (required by build_all_indices.py) ─────────────
    def save(self, path: Path) -> None:
        """Save the cache to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        logger.info("ParentCache saved to %s (%d chunks)", path, len(self.chunks))

    @classmethod
    def load(cls, path: Path) -> "ParentCache":
        """Load a cache from a JSON file."""
        cache = cls()
        with open(path, "r", encoding="utf-8") as f:
            cache.chunks = json.load(f)
        # Restore counter to avoid ID collisions if adding more later
        if cache.chunks:
            max_id = max(int(k.split("_")[1]) for k in cache.chunks.keys())
            cache._counter = max_id + 1
        logger.info("ParentCache loaded from %s (%d chunks)", path, len(cache.chunks))
        return cache


# ── Standalone builder function (required by build_all_indices.py) ─────────
def build_parent_cache(raw_data: dict) -> ParentCache:
    """
    Convenience wrapper: creates a ParentCache and loads it from raw dataset.
    
    Args:
        raw_data: Parsed dict from RAG_DataSet.json
        
    Returns:
        Fully populated ParentCache instance
    """
    cache = ParentCache()
    cache.load_from_dataset(raw_data)
    return cache


# ── Minimal test (run with: python -m rag_pipeline.indexing.parent_cache) ─
if __name__ == "__main__":
    # Minimal synthetic dataset test
    mock_data = {
        "IN-AIGOV-001": {
            "knowledge_chunks": [
                {
                    "topic": "vendor lock-in",
                    "beginner_explanation": "Vendor lock-in is dependency on one supplier.",
                    "operational_explanation": "Operationally, it restricts switching.",
                    "strategic_explanation": "Strategically it undermines sovereignty.",
                    "detailed_explanation": "Full dependency on proprietary tech.",
                    "difficulty_level": "all",
                }
            ]
        },
        "beginner_terms": [
            {"term": "AI", "simple_explanation": "Artificial Intelligence."}
        ],
        "meta": {"version": "1.0"}
    }
    
    # Test build_parent_cache function
    cache = build_parent_cache(mock_data)
    assert len(cache) == 2, f"Expected 2 chunks, got {len(cache)}"
    
    chunk = cache.get("chunk_0000")
    assert chunk["scenario_id"] == "IN-AIGOV-001"
    assert chunk["topic"] == "vendor lock-in"
    
    # Test save/load roundtrip
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = Path(f.name)
    
    cache.save(temp_path)
    loaded = ParentCache.load(temp_path)
    assert len(loaded) == 2
    assert loaded.get("chunk_0000")["topic"] == "vendor lock-in"
    
    # Cleanup
    temp_path.unlink()
    
    print("✅ parent_cache tests passed")