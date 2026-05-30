# FILE: rag_pipeline/context/compressor.py
"""
Tiered contextual compressor: selects level-appropriate fields from retrieved chunks.
Produces concise, officer-level-tailored context for prompt injection.
"""

import logging
import re
from typing import List, Dict, Tuple, Optional

from ..config import CONTEXT_BUDGET_CHARS, LEVEL_FIELD_MAP, STYLE_GUIDE
from ..utils.helpers import _clean

__all__ = ["compress_for_officer"]

logger = logging.getLogger(__name__)


def compress_for_officer(
    hits: List[Dict],
    raptor_hits: List[Tuple[str, float]],
    officer_profile: Dict,
    parent_cache,
    raptor_summaries: Optional[Dict] = None,  # Optional for backward compat
    budget_chars: int = CONTEXT_BUDGET_CHARS
) -> Tuple[str, List[Dict]]:
    """
    Compress retrieved propositions into level-appropriate context.
    
    Args:
        hits: List of proposition hit dicts from retriever.search()
        raptor_hits: List of (summary_id, score) tuples from RAPTOR search
        officer_profile: Dict with 'level', 'domain', 'preferred_name', etc.
        parent_cache: ParentCache instance for chunk metadata lookup
        raptor_summaries: Optional dict of RAPTOR metadata (legacy param)
        budget_chars: Max characters for compressed context
        
    Returns:
        Tuple of:
        - compressed_context: String ready for prompt injection
        - citations: List of citation dicts for attribution
    """
    level = officer_profile.get("level", "mid")
    blocks = []
    citations = []
    
    # ── Process proposition-level hits ───────────────────────────────────────
    seen_parents = set()
    
    for hit in hits:
        parent_id = hit.get("parent_id")
        if not parent_id or parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)
        
        # Get parent chunk metadata
        parent = parent_cache.get(parent_id)
        if not parent:
            continue
            
        scenario_id = parent.get("scenario_id", "?")
        topic = parent.get("topic", "?")
        
        # Select level-appropriate explanation field
        level_field = LEVEL_FIELD_MAP.get(level, "detailed_explanation")
        main_text = _clean(parent.get(level_field) or parent.get("detailed_explanation") or "")
        
        if not main_text:
            continue
        
        # Build level-specific content blocks
        if level == "beginner":
            # Beginner: simple explanation + example + reflection
            extra = _clean(parent.get("government_example") or parent.get("why_it_matters") or "")
            refl = _clean(parent.get("reflection_questions") or "")
            
            selected = f"{main_text}"
            if extra:
                selected += f"\n📌 Example: {extra}"
            if refl:
                selected += f"\n🤔 Reflect: {refl[:200]}"
                
        elif level == "mid":
            # Mid: operational explanation + pitfalls + reflection
            confuse = _clean(parent.get("possible_officer_confusions") or "")
            refl = _clean(parent.get("reflection_questions") or "")
            
            selected = f"{main_text}"
            if confuse:
                selected += f"\n⚠️ Watch out: {confuse}"
            if refl:
                selected += f"\n🤔 Reflect: {refl[:200]}"
                
        else:  # senior
            # Senior: strategic explanation + legal basis + governance
            laws = _clean(parent.get("relevant_laws") or "")
            govp = _clean(parent.get("governance_principles") or "")
            
            selected = f"{main_text}"
            if laws:
                selected += f"\n⚖️ Legal basis: {laws}"
            if govp:
                selected += f"\n🏛️ Governance: {govp}"
        
        # Format with citation
        citation_label = f"[Scenario: {scenario_id} | Topic: {topic}]"
        block = f"{citation_label}\n{selected}\n{'─' * 60}"
        
        blocks.append(block)
        citations.append({
            "scenario_id": scenario_id,
            "topic": topic,
            "chunk_id": parent_id,
            "level": level
        })
    
    # ── Append RAPTOR summary context (for broad/comparative queries) ────────
    if raptor_hits and raptor_summaries:
        for summary_id, score in raptor_hits:
            raptor_obj = raptor_summaries.get(summary_id)
            if not raptor_obj:
                continue
                
            rtext = _clean(raptor_obj.get("text", "")[:800])
            if rtext:
                level_label = raptor_obj.get("level", 1)
                blocks.append(
                    f"[RAPTOR Summary | Level {level_label}]\n{rtext}\n{'─' * 60}"
                )
    
    # ── Apply character budget limit ─────────────────────────────────────────
    compressed = ""
    for block in blocks:
        if len(compressed) + len(block) + 2 > budget_chars:
            logger.info(f"⚠️ Context truncated at {len(compressed)}/{budget_chars} chars")
            break
        compressed += block + "\n\n"
    
    return compressed.strip(), citations


def _extract_reflection_question(answer_text: str) -> Optional[str]:
    """
    Extract the reflection question from a generated answer.
    
    Args:
        answer_text: Full answer string from LLM
        
    Returns:
        Reflection question string or None if not found
    """
    # Pattern 1: Explicit "Reflect:" or "Reflection question:" markers
    patterns = [
        r'(?:Reflect(?:ion)?\s*[Qq]uestion[s]?:?\s*)(.+?)(?:\n|$)',
        r'(?:🤔\s*)(.+?\?)',
        r'(?:Consider:?\s*)(.+?\?)',
        r'(?:What\s+if.+?\?)',
        r'(?:How\s+would.+?\?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, answer_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip('?') + '?'
    
    # Pattern 2: Last sentence if it ends with question mark
    sentences = re.split(r'(?<=[.!?])\s+', answer_text.strip())
    if sentences and sentences[-1].endswith('?'):
        return sentences[-1].strip()
    
    return None


# ── Test ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Mock parent cache for testing
    class MockCache:
        def __init__(self):
            self._chunks = {
                "chunk_0000": {
                    "scenario_id": "IN-AIGOV-001",
                    "topic": "vendor lock-in",
                    "beginner_explanation": "Vendor lock-in means depending on one supplier.",
                    "operational_explanation": "Operationally, switching vendors incurs high costs.",
                    "strategic_explanation": "Strategically, lock-in undermines data sovereignty.",
                    "government_example": "NIC used a single cloud provider for 5 years.",
                    "possible_officer_confusions": "Confusing SaaS with managed services.",
                    "relevant_laws": "DPDP Act 2023 Section 4 requires consent.",
                    "governance_principles": "Transparency, accountability, proportionality.",
                    "reflection_questions": "What alternatives could reduce dependency?",
                }
            }
        
        def get(self, key, default=None):
            return self._chunks.get(key, default)
    
    # Mock hits
    mock_hits = [
        {
            "parent_id": "chunk_0000",
            "scenario_id": "IN-AIGOV-001",
            "topic": "vendor lock-in",
            "rrf_score": 0.95
        }
    ]
    
    mock_raptor_hits = []
    mock_profile = {"level": "mid", "domain": "procurement"}
    mock_cache = MockCache()
    
    # Test compression for each level
    for level in ["beginner", "mid", "senior"]:
        profile = {**mock_profile, "level": level}
        context, citations = compress_for_officer(
            hits=mock_hits,
            raptor_hits=mock_raptor_hits,
            officer_profile=profile,
            parent_cache=mock_cache,
            budget_chars=2000
        )
        
        assert len(context) > 0, f"Context should not be empty for {level}"
        assert len(citations) == 1, f"Should have 1 citation for {level}"
        assert citations[0]["topic"] == "vendor lock-in"
        
        print(f"✅ Level {level}: {len(context)} chars, {len(citations)} citations")
    
    print("✅ compressor tests passed")