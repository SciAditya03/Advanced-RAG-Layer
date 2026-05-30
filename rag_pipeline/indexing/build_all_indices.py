# FILE: rag_pipeline/indexing/build_all_indices.py
"""
Build all retrieval indices from raw RAG_DataSet.json
Usage: python -m rag_pipeline.indexing.build_all_indices
"""

import json, logging, pickle, time, os
from pathlib import Path
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Import internal modules
from ..config import (
    RAW_DATA_PATH, PROCESSED_DIR, INDICES_DIR,
    PROPOSITIONS_PATH, PARENT_CACHE_PATH,
    FAISS_INDEX_PATH, BM25_INDEX_PATH,
    RAPTOR_FAISS_PATH, RAPTOR_METADATA_PATH, INDEX_MANIFEST_PATH,
    EMBEDDING_MODEL, DEVICE, FAISS_DIM, RRF_K
)
from ..utils.helpers import _clean, safe_get, tokenize_for_bm25
from .parent_cache import build_parent_cache, ParentCache
from .proposition_generator import generate_all_propositions
from .faiss_builder import build_faiss_index, save_faiss_index
from .bm25_builder import build_bm25_index, save_bm25_index, tokenize_for_bm25
from .raptor_builder import build_raptor_hierarchy, save_raptor_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def ensure_dirs():
    """Create output directories if they don't exist."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Ensured directories: {PROCESSED_DIR}, {INDICES_DIR}")


def load_raw_dataset(path: Path) -> dict:
    """Load and validate the raw JSON dataset."""
    if not path.exists():
        raise FileNotFoundError(f"❌ Dataset not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    logger.info(f"✅ Loaded dataset: {len(data)} top-level entries")
    return data


def save_manifest(stats: dict):
    """Save index build metadata for versioning."""
    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_path": str(RAW_DATA_PATH),
        "embedding_model": EMBEDDING_MODEL,
        "device": DEVICE,
        **stats
    }
    with open(INDEX_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"📋 Manifest saved: {INDEX_MANIFEST_PATH}")


def main():
    """Main index building pipeline."""
    t0 = time.time()
    logger.info("🚀 Starting index build pipeline...")
    
    # Ensure output directories exist
    ensure_dirs()
    
    # 1. Load raw dataset
    logger.info("📦 Step 1/6: Loading raw dataset...")
    raw_data = load_raw_dataset(RAW_DATA_PATH)
    
    # 2. Build parent chunk cache
    logger.info("🗂️  Step 2/6: Building parent chunk cache...")
    parent_cache = build_parent_cache(raw_data)
    parent_cache.save(PARENT_CACHE_PATH)
    logger.info(f"   → Saved {len(parent_cache)} parent chunks to {PARENT_CACHE_PATH}")
    
    # 3. Generate propositions
    logger.info("🔍 Step 3/6: Generating propositions...")
    propositions = generate_all_propositions(parent_cache)
    with open(PROPOSITIONS_PATH, "w", encoding="utf-8") as f:
        for prop in propositions:
            f.write(json.dumps(prop, ensure_ascii=False) + "\n")
    logger.info(f"   → Saved {len(propositions)} propositions to {PROPOSITIONS_PATH}")
    
    # 4. Build FAISS index (dense embeddings)
    logger.info("🧠 Step 4/6: Building FAISS index...")
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    faiss_index = build_faiss_index(propositions, embedder, DEVICE)
    save_faiss_index(faiss_index, FAISS_INDEX_PATH)
    logger.info(f"   → Saved FAISS index: {faiss_index.ntotal} vectors, dim={FAISS_DIM}")
    
    # 5. Build BM25 index (sparse keyword search)
    logger.info("📚 Step 5/6: Building BM25 index...")
    # Build BM25 index - returns BM25Okapi object
    bm25_index = build_bm25_index(propositions)
    # Track document count separately (BM25Okapi doesn't expose corpus)
    bm25_doc_count = len(propositions)
    save_bm25_index(bm25_index, BM25_INDEX_PATH)
    logger.info(f"   → Saved BM25 index: {bm25_doc_count} documents")
    
    # 6. Build RAPTOR hierarchy (broad-query summaries)
    logger.info("🌲 Step 6/6: Building RAPTOR summary hierarchy...")
    raptor_faiss, raptor_metadata = build_raptor_hierarchy(
        parent_cache, propositions, embedder, DEVICE
    )
    save_raptor_data(raptor_metadata, raptor_faiss, INDICES_DIR)
    logger.info(f"   → Saved RAPTOR: {len(raptor_metadata)} nodes")
    
    # Save manifest
    stats = {
        "parent_chunks": len(parent_cache),
        "propositions": len(propositions),
        "faiss_vectors": faiss_index.ntotal,
        "bm25_docs": bm25_doc_count,
        "raptor_nodes": len(raptor_metadata)
    }
    save_manifest(stats)
    
    elapsed = time.time() - t0
    logger.info(f"✅ Index build complete in {elapsed:.1f}s")
    logger.info(f"📁 All indices saved to: {INDICES_DIR}")
    logger.info("🎯 Next: Test with python -m rag_pipeline.pipeline.governance_rag --demo")


if __name__ == "__main__":
    main()