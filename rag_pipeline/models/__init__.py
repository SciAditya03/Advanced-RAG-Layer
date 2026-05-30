# FILE: rag_pipeline/models/__init__.py
"""
Models module: LLM wrappers, verification, and generation utilities.
"""

from .llm_wrapper import generate_answer, load_llm, LLM_AVAILABLE
from .crag_verifier import verify_answer, CRAG_PROMPT
from .embedder import EmbedderWrapper, load_embedder

__all__ = [
    # LLM generation
    "generate_answer",
    "load_llm", 
    "LLM_AVAILABLE",
    
    # Verification
    "verify_answer",
    "CRAG_PROMPT",
    
    # Embeddings
    "EmbedderWrapper",
    "load_embedder",
]