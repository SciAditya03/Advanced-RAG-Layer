"""
Rule-based query preprocessor: intent classification, keyword expansion,
scenario filtering, and sub-query splitting for comparative intents.
"""

import re
import logging

__all__ = ["preprocess_query", "classify_intent", "expand_keywords_from_dataset"]

logger = logging.getLogger(__name__)

from ..config import INTENT_KEYWORDS
from ..utils.helpers import tokenize_for_bm25


def classify_intent(query: str) -> str:
    """Classify query intent from keyword patterns.

    Args:
        query: Raw query string.

    Returns:
        One of 'specific', 'comparative', or 'general'.

    Example:
        >>> classify_intent("compare vendor lock-in vs open source")
        'comparative'
    """
    q = query.lower()
    for intent, kws in INTENT_KEYWORDS.items():
        if any(kw in q for kw in kws):
            return intent
    return "specific"


def expand_keywords_from_dataset(query: str, parent_chunks: dict,
                                  top_n: int = 5) -> list[str]:
    """Find dataset retrieval_keywords that overlap with query tokens.

    Args:
        query: Raw query string.
        parent_chunks: dict[chunk_id → metadata] from ParentChunkCache.
        top_n: Maximum number of expanded keywords to return.

    Returns:
        List of extra keyword strings.
    """
    q_tokens = set(tokenize_for_bm25(query))
    expanded: list[str] = []

    for chunk in parent_chunks.values():
        kws = chunk.get("retrieval_keywords")
        if isinstance(kws, list):
            kw_tokens = set(w.lower() for kw in kws for w in kw.split())
        else:
            kw_tokens = set(tokenize_for_bm25(str(kws or "")))

        if q_tokens & kw_tokens:
            expanded += list(kw_tokens - q_tokens)

    # Deduplicate
    seen: set[str] = set()
    result: list[str] = []
    for kw in expanded:
        if kw and kw not in seen:
            seen.add(kw)
            result.append(kw)

    return result[:top_n]


def preprocess_query(query: str, officer_profile: dict,
                     parent_chunks: dict | None = None,
                     scenario_ids: list[str] | None = None) -> dict:
    """Full query preprocessing pipeline.

    Rewrites query with officer-level hint, classifies intent, infers
    scenario filter, splits comparative queries, and expands keywords.

    Args:
        query: Officer's raw question.
        officer_profile: Dict with at least a 'level' key.
        parent_chunks: Optional parent chunk dict for keyword expansion.
        scenario_ids: Optional list of known scenario IDs for filter inference.

    Returns:
        Dict with keys:
            rewritten_query : str
            intent          : str ('specific' | 'comparative' | 'general')
            filters         : dict (level, optionally scenario_id)
            sub_queries     : list[str]
            keywords        : list[str]

    Example:
        >>> preprocess_query("What is vendor lock-in?", {"level": "mid"})
        {'rewritten_query': ..., 'intent': 'specific', ...}
    """
    level = officer_profile.get("level", "mid")

    rewritten = f"{query} [Officer level: {level}]"
    intent    = classify_intent(query)

    # Scenario filter from explicit mention
    scenario_filter = None
    if scenario_ids:
        for sid in scenario_ids:
            if sid.replace("_", " ").lower() in query.lower() or sid.lower() in query.lower():
                scenario_filter = sid
                break

    filters: dict = {"level": level}
    if scenario_filter:
        filters["scenario_id"] = scenario_filter

    # Sub-queries for comparative intent
    sub_queries: list[str] = []
    if intent == "comparative":
        parts = re.split(
            r'\b(?:vs\.?|versus|compare|difference between|and)\b', query, flags=re.I
        )
        sub_queries = [p.strip() for p in parts if len(p.strip()) > 3]

    if not sub_queries:
        sub_queries = [query]

    # Keyword expansion
    keywords: list[str] = []
    if parent_chunks:
        keywords = expand_keywords_from_dataset(query, parent_chunks)

    result = {
        "rewritten_query": rewritten,
        "intent":          intent,
        "filters":         filters,
        "sub_queries":     sub_queries,
        "keywords":        keywords,
    }

    logger.debug("preprocess_query: intent=%s filters=%s kws=%s",
                 intent, filters, keywords)
    return result


if __name__ == "__main__":
    # Basic intent tests
    assert classify_intent("compare open source vs proprietary AI") == "comparative"
    assert classify_intent("what is vendor lock-in") == "specific"
    assert classify_intent("overview of AI governance principles") == "general"

    # Preprocess without chunks
    r = preprocess_query("What are the risks of free AI vendors?", {"level": "mid"})
    assert r["intent"] == "specific"
    assert r["filters"]["level"] == "mid"
    assert r["sub_queries"] == ["What are the risks of free AI vendors?"]

    # Comparative splitting
    r2 = preprocess_query("compare open source vs proprietary", {"level": "senior"})
    assert r2["intent"] == "comparative"
    assert len(r2["sub_queries"]) >= 2

    print("✅ query_preprocessor tests passed")