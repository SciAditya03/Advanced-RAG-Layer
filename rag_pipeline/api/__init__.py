"""FastAPI wrapper exposing the RAG pipeline as a microservice."""

__all__ = ["app"]

from .rag_service import app