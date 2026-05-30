"""
Abstract base interface for all RAG pipeline variants.
Enforces the run() contract so alternative pipelines remain interchangeable.
"""

from abc import ABC, abstractmethod

__all__ = ["BaseRAGPipeline"]


class BaseRAGPipeline(ABC):
    """Abstract RAG pipeline interface.

    Concrete implementations must override run().
    """

    @abstractmethod
    def run(
        self,
        query: str,
        officer_profile: dict,
        chat_history: list[dict] | None = None,
        scenario_id: str | None = None,
        max_retries: int = 1,
    ) -> dict:
        """Execute the full RAG pipeline and return a result dict.

        Args:
            query: Officer's natural-language question.
            officer_profile: Dict with keys: level, domain, preferred_name, …
            chat_history: Conversation history for multi-turn rewriting.
            scenario_id: Optional — restrict retrieval to one scenario.
            max_retries: CRAG retry limit.

        Returns:
            Dict with at minimum these keys:
                answer, citations, verification_log, reflection_question,
                retrieved_propositions, compressed_context, rewritten_query, elapsed_s
        """
        ...