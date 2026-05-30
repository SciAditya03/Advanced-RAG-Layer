"""
Configuration: paths, model settings, retrieval hyperparameters.
All tunable constants live here; never hard-code in other modules.
"""

from pathlib import Path
import torch
import logging

logger = logging.getLogger(__name__)

__all__ = [
    # Paths
    "BASE_DIR", "INDICES_DIR", "DATA_DIR",
    "RAW_DATA_PATH", "PROCESSED_DIR", "PROPOSITIONS_PATH", "PARENT_CACHE_PATH",
    "FAISS_INDEX_PATH", "BM25_INDEX_PATH", "RAPTOR_FAISS_PATH", "RAPTOR_METADATA_PATH", "INDEX_MANIFEST_PATH",
    # Models
    "DEVICE", "EMBEDDING_MODEL", "LLM_MODEL",
    # Retrieval params
    "FAISS_DIM", "BM25_K1", "BM25_B", "RRF_K",
    "TOP_K_PROPOSITIONS", "TOP_K_RAPTOR",
    # Context
    "CONTEXT_BUDGET_CHARS", "LEVEL_FIELD_MAP",
    # Prompts
    "STYLE_GUIDE", "INTENT_KEYWORDS", "SCENARIO_SKIP_KEYS",
    # Performance
    "PRELOAD_EMBEDDER", "PRELOAD_LLM", "EMBEDDER_LOAD_TIMEOUT", "LLM_LOAD_TIMEOUT",
    "QUERY_EXECUTION_TIMEOUT", "USE_ONNX_RUNTIME", "ENABLE_MODEL_QUANTIZATION",
    "ENABLE_QUERY_CACHE", "QUERY_CACHE_TTL_SECONDS", "PERF_LOG_LEVEL",
    "BM25_ONLY_MODE", "FORCE_CPU_MODE", "USE_LLM",
]

# ── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
INDICES_DIR = BASE_DIR / "indices"

# ── Raw & Processed Data Paths ────────────────────────────────────────────────
RAW_DATA_PATH       = DATA_DIR / "raw" / "RAG_DataSet.json"
PROCESSED_DIR       = DATA_DIR / "processed"
PROPOSITIONS_PATH   = PROCESSED_DIR / "propositions.jsonl"
PARENT_CACHE_PATH   = PROCESSED_DIR / "parent_cache.json"

# ── Index File Paths ──────────────────────────────────────────────────────────
FAISS_INDEX_PATH    = INDICES_DIR / "faiss_index.bin"
BM25_INDEX_PATH     = INDICES_DIR / "bm25_index.pkl"
RAPTOR_FAISS_PATH   = INDICES_DIR / "raptor_faiss.bin"
RAPTOR_METADATA_PATH= INDICES_DIR / "raptor_metadata.json"
INDEX_MANIFEST_PATH = INDICES_DIR / "index_manifest.json"

# ── Model settings ────────────────────────────────────────────────────────────
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL       = "Qwen/Qwen2.5-1.5B-Instruct"

# ── Index dimensions ──────────────────────────────────────────────────────────
FAISS_DIM = 384    # matches all-MiniLM-L6-v2 output dim

# ── BM25 hyperparameters ──────────────────────────────────────────────────────
BM25_K1 = 1.5
BM25_B  = 0.75

# ── RRF merge constant ────────────────────────────────────────────────────────
RRF_K = 60

# ── Retrieval top-k defaults ──────────────────────────────────────────────────
TOP_K_PROPOSITIONS = 10
TOP_K_RAPTOR       = 2

# ── Context compression budget ────────────────────────────────────────────────
CONTEXT_BUDGET_CHARS = 6000

# ── Level → field name mapping ────────────────────────────────────────────────
LEVEL_FIELD_MAP = {
    "beginner": "beginner_explanation",
    "mid":      "operational_explanation",
    "senior":   "strategic_explanation",
}

# ── Teaching style guide per level ───────────────────────────────────────────
STYLE_GUIDE = {
    "beginner": (
        "Use simple language and short sentences. "
        "Give one concrete government example. "
        "Avoid technical jargon. End with a reflection question."
    ),
    "mid": (
        "Focus on operational procedures and workflow implications. "
        "Highlight common pitfalls. "
        "Be actionable and direct. End with a reflection question."
    ),
    "senior": (
        "Emphasise strategic trade-offs, policy architecture, and legal frameworks. "
        "Reference relevant laws and governance principles. "
        "Connect to broader AI governance goals. End with a reflection question."
    ),
}

# ── Intent classification keywords ───────────────────────────────────────────
INTENT_KEYWORDS = {
    "specific":    ["how to", "what is", "define", "explain", "procedure", "steps"],
    "comparative": ["compare", "difference", "vs", "versus", "which", "better"],
    "general":     ["overview", "principles", "summary", "all", "main", "key"],
}

# ── Keys to skip when extracting scenario IDs from dataset ───────────────────
SCENARIO_SKIP_KEYS = {"meta", "beginner_terms"}

# ── PERFORMANCE & LATENCY SETTINGS ───────────────────────────────────────────
# Pre-load heavy components at startup (True = slower init, faster queries)
PRELOAD_EMBEDDER = False  # Set True after embedder is verified working
PRELOAD_LLM = False       # Set True only if you have 4GB+ RAM

# Timeout protection (seconds) — prevents infinite hangs
EMBEDDER_LOAD_TIMEOUT = 30
LLM_LOAD_TIMEOUT = 60
QUERY_EXECUTION_TIMEOUT = 15

# Model optimization flags
USE_ONNX_RUNTIME = False  # Set True for 2-3x faster inference (requires onnxruntime)
ENABLE_MODEL_QUANTIZATION = True  # Use 4-bit/8-bit models when available

# Cache settings
ENABLE_QUERY_CACHE = True
QUERY_CACHE_TTL_SECONDS = 300  # Cache identical queries for 5 minutes

# Logging level for performance monitoring
PERF_LOG_LEVEL = "INFO"  # "DEBUG" for detailed timing, "INFO" for summary

# ── TESTING/DEVELOPMENT MODES ────────────────────────────────────────────────
# Set to True to skip FAISS/embedder loading (BM25 keyword search only)
BM25_ONLY_MODE = True  # ← Keep True for fast testing; set False when ready

# Force CPU for all model loading (avoid CUDA issues on Windows)
FORCE_CPU_MODE = True

# Enable LLM generation (set False for context-echo fallback mode)
USE_LLM = False  # ← Keep False for fast testing; set True when ready

# ── Apply FORCE_CPU_MODE if configured ───────────────────────────────────────
if FORCE_CPU_MODE and DEVICE == "cuda":
    DEVICE = "cpu"
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    logger.warning("⚠️  Forced CPU mode (set FORCE_CPU_MODE=False in config.py to enable CUDA)")