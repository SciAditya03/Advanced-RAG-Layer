# FILE: rag_pipeline/indexing/__init__.py
"""
Indexing module: builds FAISS, BM25, RAPTOR indices from raw dataset.
"""

from .parent_cache import ParentCache, build_parent_cache
from .proposition_generator import generate_all_propositions, build_propositions_for_chunk
from .faiss_builder import build_faiss_index, save_faiss_index, load_faiss_index
from .bm25_builder import (
    build_bm25_index, 
    save_bm25_index, 
    load_bm25_index, 
    bm25_retrieve,
    tokenize_for_bm25
)
from .raptor_builder import (
    build_raptor_hierarchy, 
    save_raptor_data, 
    load_raptor_data
)

__all__ = [
    # Parent cache
    "ParentCache",
    "build_parent_cache",
    
    # Propositions
    "generate_all_propositions",
    "build_propositions_for_chunk",
    
    # FAISS
    "build_faiss_index",
    "save_faiss_index", 
    "load_faiss_index",
    
    # BM25
    "build_bm25_index",
    "save_bm25_index",
    "load_bm25_index",
    "bm25_retrieve",
    "tokenize_for_bm25",
    
    # RAPTOR
    "build_raptor_hierarchy",
    "save_raptor_data",
    "load_raptor_data",
]