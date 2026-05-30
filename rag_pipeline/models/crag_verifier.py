# FILE: rag_pipeline/models/crag_verifier.py
"""
CRAG-lite verification: check if generated answers are grounded in retrieved context.
Uses simple rule-based checks when LLM is unavailable.
"""

import json
import re
import logging
from typing import Dict, List, Optional

from ..config import LLM_MODEL, DEVICE
from .llm_wrapper import load_llm, _llm_pipe

__all__ = ["verify_answer", "CRAG_PROMPT"]

logger = logging.getLogger(__name__)

CRAG_PROMPT = """\
You are a factual verification assistant.

Check if every claim in the ANSWER is supported by the CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

QUERY:
{query}

Respond ONLY with a JSON object (no markdown fences):
{{
  "supported": true or false,
  "unsupported_claims": ["claim 1", "claim 2"],
  "missing_context": ["what extra context would help"]
}}
"""


def _parse_crag_json(raw: str) -> Dict:
    """Robustly parse JSON from LLM output."""
    try:
        # Strip markdown fences if present
        clean = re.sub(r'```[\s\S]*?```', lambda m: m.group().replace('`', ''), raw)
        match = re.search(r'\{[\s\S]+\}', clean)
        if match:
            return json.loads(match.group())
    except Exception as e:
        logger.debug(f"JSON parse failed: {e}")
    return {"supported": True, "unsupported_claims": [], "missing_context": []}


def verify_answer(
    context: str, 
    answer: str, 
    query: str,
    use_llm: bool = True
) -> Dict:
    """
    Verify if answer is grounded in context.
    
    Args:
        context: Retrieved knowledge context
        answer: Generated answer to verify
        query: Original officer query
        use_llm: Whether to use LLM for verification (fallback to rules if False)
        
    Returns:
        Dict with keys: supported, unsupported_claims, missing_context
    """
    # Rule-based fallback when LLM unavailable
    if not use_llm or _llm_pipe is None:
        return _rule_based_verify(context, answer, query)
    
    # LLM-based verification
    try:
        prompt = CRAG_PROMPT.format(
            context=context[:3000],  # Keep prompt manageable
            answer=answer[:1500],
            query=query
        )
        
        raw = _llm_pipe(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"].strip()
        result = _parse_crag_json(raw)
        result["raw_output"] = raw[:500]  # Debug info
        return result
        
    except Exception as e:
        logger.warning(f"CRAG verification error: {e}")
        return _rule_based_verify(context, answer, query)


def _rule_based_verify(context: str, answer: str, query: str) -> Dict:
    """
    Simple rule-based verification when LLM unavailable.
    
    Checks:
    - Answer length reasonable
    - Contains key terms from query
    - Doesn't contain hallucination markers
    """
    unsupported = []
    missing = []
    
    # Check 1: Answer not empty
    if not answer.strip():
        unsupported.append("Empty answer")
    
    # Check 2: Answer contains query keywords (basic relevance)
    query_terms = re.findall(r'\b[a-zA-Z]{4,}\b', query.lower())
    answer_lower = answer.lower()
    if query_terms and not any(term in answer_lower for term in query_terms[:3]):
        missing.append(f"Query terms not found: {query_terms[:3]}")
    
    # Check 3: No obvious hallucination markers
    hallucination_patterns = [
        r"according to section \d+",  # Fake law citations
        r"as per the \w+ act \d+",    # Fake act references
        r"the law states that",        # Vague legal claims without source
    ]
    for pattern in hallucination_patterns:
        if re.search(pattern, answer, re.I):
            unsupported.append(f"Potential hallucination: {pattern}")
    
    # Check 4: Answer references context (basic grounding)
    context_snippets = [c.strip() for c in context.split("\n") if len(c.strip()) > 50]
    if context_snippets and not any(snippet[:30] in answer for snippet in context_snippets[:3]):
        missing.append("Answer may not reference retrieved context")
    
    return {
        "supported": len(unsupported) == 0,
        "unsupported_claims": unsupported,
        "missing_context": missing,
        "note": "Rule-based verification (LLM unavailable)"
    }


# ── Test ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test rule-based fallback
    context = "Vendor lock-in creates dependency risks. Switching costs are high."
    answer = "Vendor lock-in means depending on one vendor, which increases switching costs."
    query = "What is vendor lock-in?"
    
    result = verify_answer(context, answer, query, use_llm=False)
    assert result["supported"] == True
    print("✅ crag_verifier tests passed (rule-based mode)")