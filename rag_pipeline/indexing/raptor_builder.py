"""
RAPTOR 2-level summary hierarchy builder.
Level 1: one summary per scenario.
Level 2: one cross-scenario governance synthesis.
No external LLM required — uses template-based concatenation.
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

from ..utils.helpers import _clean

__all__ = ["build_raptor_hierarchy", "save_raptor_data", "load_raptor_data"]

logger = logging.getLogger(__name__)


def _build_summary_text(chunks: List[dict], level_key: str = "detailed_explanation") -> str:
    """Concatenate key explanations from a list of parent chunks."""
    parts = []
    for c in chunks:
        topic = c.get("topic", "")
        text = _clean(c.get(level_key) or c.get("detailed_explanation") or "")
        if text:
            parts.append(f"[{topic}] {text[:400]}")
    return "\n\n".join(parts)


def build_raptor_hierarchy(
    parent_cache,
    propositions: List[Dict],
    embedder,
    device: str
) -> Tuple:
    """Build a 2-level RAPTOR hierarchy and FAISS index over it.

    Args:
        parent_cache: ParentCache instance with .chunks attribute
        propositions: List of proposition dicts (for metadata extraction)
        embedder: Embedder instance for computing summary embeddings
        device: "cuda" or "cpu"

    Returns:
        Tuple of:
            raptor_faiss     : faiss.IndexFlatIP
            raptor_metadata  : dict[summary_id → {text, level, child_ids, scenario_id}]
    """
    import faiss

    # Accept both ParentCache object or raw dict
    chunks = parent_cache.chunks if hasattr(parent_cache, "chunks") else parent_cache
    
    raptor_summaries: Dict[str, dict] = {}
    scenario_summary_ids: List[str] = []

    # Extract unique scenario IDs (excluding meta/beginner_terms)
    scenario_ids = list(set(
        c["scenario_id"] for c in chunks.values() 
        if c["scenario_id"] not in {"meta", "beginner_terms"}
    ))

    # ── Level 1: per-scenario summaries ──────────────────────────────────────
    for sid in scenario_ids:
        scenario_chunks = [c for c in chunks.values() if c["scenario_id"] == sid]
        if not scenario_chunks:
            continue
        summary_text = (
            f"[Scenario Summary: {sid}]\n"
            + _build_summary_text(scenario_chunks, "detailed_explanation")
        )
        smid = f"raptor_L1_{sid}"
        raptor_summaries[smid] = {
            "text":        summary_text,
            "level":       1,
            "child_ids":   [c["chunk_id"] for c in scenario_chunks],
            "scenario_id": sid,
        }
        scenario_summary_ids.append(smid)

    # ── Level 2: global cross-scenario summary ────────────────────────────────
    if scenario_summary_ids:
        all_l1_texts = "\n\n".join(
            raptor_summaries[smid]["text"][:500] for smid in scenario_summary_ids
        )
        global_summary = (
            "[Cross-Scenario Governance Summary]\n"
            "The following governance principles recur across scenarios:\n"
            + all_l1_texts[:3000]
        )
        raptor_summaries["raptor_L2_global"] = {
            "text":        global_summary,
            "level":       2,
            "child_ids":   scenario_summary_ids,
            "scenario_id": "global",
        }

    # ── Embed all RAPTOR nodes ─────────────────────────────────────────────────
    raptor_ids: List[str] = list(raptor_summaries.keys())
    raptor_texts: List[str] = [raptor_summaries[smid]["text"] for smid in raptor_ids]

    if raptor_texts:
        embs = embedder.encode(
            raptor_texts, 
            normalize_embeddings=True, 
            show_progress_bar=False,
            convert_to_numpy=True
        ).astype("float32")
        
        dim = embs.shape[1]
        raptor_faiss = faiss.IndexFlatIP(dim)
        raptor_faiss.add(embs)
    else:
        # Empty fallback index
        raptor_faiss = faiss.IndexFlatIP(384)

    logger.info(
        "RAPTOR index built: %d nodes (L1=%d, L2=1)",
        len(raptor_summaries), len(scenario_summary_ids)
    )
    return raptor_faiss, raptor_summaries


def save_raptor_data(raptor_summaries: dict, raptor_faiss, dir_path: str | Path) -> None:
    """Persist RAPTOR artefacts to a directory."""
    import faiss, json
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    faiss.write_index(raptor_faiss, str(dir_path / "raptor_faiss.bin"))
    with open(dir_path / "raptor_metadata.json", "w", encoding="utf-8") as f:
        json.dump(raptor_summaries, f, indent=2, ensure_ascii=False)
    
    logger.info("RAPTOR data saved to %s", dir_path)


def load_raptor_data(dir_path: str | Path) -> Tuple:
    """Load RAPTOR artefacts from disk."""
    import faiss, json
    dir_path = Path(dir_path)

    def _require(p: Path):
        if not p.exists():
            raise FileNotFoundError(f"Missing RAPTOR file: {p}")
        return p

    raptor_faiss = faiss.read_index(str(_require(dir_path / "raptor_faiss.bin")))
    with open(_require(dir_path / "raptor_metadata.json"), "r", encoding="utf-8") as f:
        raptor_summaries = json.load(f)

    logger.info("RAPTOR data loaded from %s", dir_path)
    return raptor_faiss, raptor_summaries


if __name__ == "__main__":
    import sys
    from sentence_transformers import SentenceTransformer
    
    # Mock embedder for testing
    class MockEmbedder:
        def encode(self, texts, normalize_embeddings=True, **kwargs):
            rng = np.random.default_rng(0)
            vecs = rng.standard_normal((len(texts), 384)).astype("float32")
            if normalize_embeddings:
                vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs

    # Mock parent cache
    class MockCache:
        def __init__(self):
            self.chunks = {
                "chunk_0000": {
                    "chunk_id": "chunk_0000", 
                    "scenario_id": "IN-AIGOV-001", 
                    "topic": "vendor lock-in",
                    "detailed_explanation": "Vendor lock-in creates dependency risks in procurement.",
                },
                "chunk_0001": {
                    "chunk_id": "chunk_0001", 
                    "scenario_id": "IN-AIGOV-002", 
                    "topic": "data privacy",
                    "detailed_explanation": "Data privacy requires explicit consent under DPDP Act.",
                },
            }

    mock_cache = MockCache()
    mock_props = [{"text": "test", "prop_id": "p1"}]
    embedder = MockEmbedder()

    raptor_faiss, metadata = build_raptor_hierarchy(mock_cache, mock_props, embedder, "cpu")
    assert "raptor_L1_IN-AIGOV-001" in metadata
    assert "raptor_L2_global" in metadata
    assert raptor_faiss.ntotal == len(metadata)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        save_raptor_data(metadata, raptor_faiss, td)
        faiss2, meta2 = load_raptor_data(td)
        assert set(meta2.keys()) == set(metadata.keys())

    print(f"✅ raptor_builder tests passed — {len(metadata)} nodes")