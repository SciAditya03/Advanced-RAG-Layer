"""
FastAPI wrapper for GovernanceRAGPipeline.
Provides the /rag/retrieve endpoint consumed by Tanya's LLM service
and the Django orchestrator.

Start:
    uvicorn rag_pipeline.api.rag_service:app --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

__all__ = ["app"]

logger = logging.getLogger(__name__)

from ..pipeline.governance_rag import GovernanceRAGPipeline
from ..config import INDICES_DIR


# ── Pydantic models ────────────────────────────────────────────────────────────

class RAGRequest(BaseModel):
    """Incoming retrieval request.

    Attributes:
        query: Officer's natural-language question.
        officer_profile: Dict with at least 'level' key.
        chat_history: Full conversation history for multi-turn rewriting.
        scenario_id: Optional scenario filter.
    """
    query:          str
    officer_profile: dict
    chat_history:   list[dict] = []
    scenario_id:    str | None = None


class RAGResponse(BaseModel):
    """Outgoing retrieval response consumed by Tanya's LLM service.

    Attributes:
        context: Compressed, level-appropriate context string.
        citations: List of source citation dicts.
        rewritten_query: Standalone query after multi-turn rewriting.
        elapsed_s: Pipeline wall-clock time.
    """
    context:         str
    citations:       list[dict]
    rewritten_query: str
    elapsed_s:       float


class HealthResponse(BaseModel):
    status: str
    pipeline_loaded: bool
    proposition_count: int


# ── Application lifecycle ──────────────────────────────────────────────────────

_pipeline: GovernanceRAGPipeline | None = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load pipeline on startup; clean up on shutdown."""
    global _pipeline
    logger.info("Loading GovernanceRAGPipeline from %s …", INDICES_DIR)
    try:
        _pipeline = GovernanceRAGPipeline(indices_path=str(INDICES_DIR))
        logger.info("Pipeline loaded successfully.")
    except Exception as exc:
        logger.error("Pipeline load failed: %s", exc)
        # Allow startup to complete so /health is still reachable
        _pipeline = None
    yield
    _pipeline = None


app = FastAPI(
    title="AI Governance RAG Service",
    description="Retrieval-augmented context service for the AI Governance Simulator.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health_check() -> HealthResponse:
    """Liveness/readiness probe."""
    return HealthResponse(
        status="ok" if _pipeline is not None else "degraded",
        pipeline_loaded=_pipeline is not None,
        proposition_count=len(_pipeline.propositions) if _pipeline else 0,
    )


@app.post("/rag/retrieve", response_model=RAGResponse, tags=["retrieval"])
def retrieve_context(req: RAGRequest) -> RAGResponse:
    """Retrieve grounded context for an officer query.

    Called by Django orchestrator before forwarding to Tanya's LLM service.

    Args:
        req: RAGRequest containing query, officer profile, history, scenario.

    Returns:
        RAGResponse with compressed context, citations, and metadata.

    Raises:
        HTTPException 503: If the pipeline was not loaded successfully.
        HTTPException 500: On unexpected pipeline errors.
    """
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "pipeline_not_loaded", "retryable": True},
        )

    try:
        result = _pipeline.run(
            query=req.query,
            officer_profile=req.officer_profile,
            chat_history=req.chat_history,
            scenario_id=req.scenario_id,
        )
    except Exception as exc:
        logger.exception("Pipeline run error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail={"error": str(exc), "retryable": False},
        )

    return RAGResponse(
        context=result["compressed_context"],
        citations=result["citations"],
        rewritten_query=result["rewritten_query"],
        elapsed_s=result["elapsed_s"],
    )