"""
Rule-based proposition generator.
Produces 3–5 atomic, level-tagged propositions per parent chunk.
No LLM required at index time.
"""

import re
import logging
from typing import List, Dict

from ..utils.helpers import _clean

__all__ = ["build_propositions_for_chunk", "generate_all_propositions"]

logger = logging.getLogger(__name__)


def build_propositions_for_chunk(chunk: dict) -> List[Dict]:
    """Generate atomic propositions from a single parent chunk.

    Creates up to 5 propositions per officer level (beginner/mid/senior),
    each prefixed with scenario/level/topic metadata for filterable retrieval.

    Args:
        chunk: Parent chunk dict (as stored in ParentCache.chunks).

    Returns:
        List of dicts with keys: text, level.
        Caller is responsible for assigning prop_id and merging metadata.
    """
    sid   = chunk["scenario_id"]
    topic = chunk["topic"]
    props = []

    level_map = {
        "beginner": chunk.get("beginner_explanation")    or chunk.get("detailed_explanation", ""),
        "mid":      chunk.get("operational_explanation") or chunk.get("detailed_explanation", ""),
        "senior":   chunk.get("strategic_explanation")   or chunk.get("detailed_explanation", ""),
    }

    for lvl, base_text in level_map.items():
        base_text = _clean(base_text)
        if not base_text:
            continue

        prefix = f"[Scenario: {sid} | Level: {lvl} | Topic: {topic}]"

        # Prop 1 & 2: first two sentences of base explanation
        sentences = re.split(r'(?<=[.!?])\s+', base_text)
        for sent in sentences[:2]:
            sent = sent.strip()
            if len(sent) > 20:
                props.append({"text": f"{prefix} {sent}", "level": lvl})

        # Prop 3: government example
        ex = _clean(chunk.get("government_example") or chunk.get("government_relevance") or "")
        if ex and len(ex) > 20:
            props.append({"text": f"{prefix} Government example: {ex[:300]}", "level": lvl})

        # Prop 4: risk / officer confusion
        risk = _clean(chunk.get("risk_types") or chunk.get("possible_officer_confusions") or "")
        if risk and len(risk) > 20:
            props.append({"text": f"{prefix} Key risk/confusion: {risk[:300]}", "level": lvl})

        # Prop 5: legal/governance (senior-relevant but available to all)
        law = _clean(chunk.get("relevant_laws") or chunk.get("governance_principles") or "")
        if law and len(law) > 20:
            props.append({"text": f"{prefix} Legal/governance basis: {law[:300]}", "level": lvl})

        # Prop 6: why it matters (beginner-friendly)
        wim = _clean(chunk.get("why_it_matters") or "")
        if wim and len(wim) > 20:
            props.append({"text": f"{prefix} Why it matters: {wim[:300]}", "level": lvl})

    return props


def generate_all_propositions(parent_cache) -> List[Dict]:
    """Generate propositions for every chunk in the parent cache.

    Args:
        parent_cache: ParentCache instance with .chunks attribute

    Returns:
        List of proposition dicts with keys:
            prop_id, text, parent_id, scenario_id, level, topic.
    """
    propositions = []
    counter = 0

    # Accept both ParentCache object or raw dict for flexibility
    chunks = parent_cache.chunks if hasattr(parent_cache, "chunks") else parent_cache

    for cid, chunk in chunks.items():
        for prop_dict in build_propositions_for_chunk(chunk):
            pid = f"prop_{counter:05d}"
            counter += 1
            propositions.append({
                "prop_id":     pid,
                "text":        prop_dict["text"],
                "parent_id":   cid,
                "scenario_id": chunk["scenario_id"],
                "level":       prop_dict["level"],
                "topic":       chunk["topic"],
            })

    logger.info("Generated %d propositions from %d chunks", len(propositions), len(chunks))
    return propositions


if __name__ == "__main__":
    mock_chunk = {
        "scenario_id": "IN-AIGOV-001",
        "topic": "vendor lock-in",
        "beginner_explanation": "Vendor lock-in means dependency on one supplier. It reduces flexibility.",
        "operational_explanation": "Operationally, switching vendors is costly and disruptive.",
        "strategic_explanation": "Strategically, lock-in undermines data sovereignty.",
        "detailed_explanation": "Vendor lock-in occurs when switching costs are prohibitively high.",
        "government_example": "NIC relied on a single cloud provider for five years.",
        "risk_types": "Budget overruns, vendor price increases.",
        "why_it_matters": "Governments must retain control of public data.",
        "relevant_laws": "IT Act 2000, PDPB 2023",
        "governance_principles": "Transparency, accountability.",
        "possible_officer_confusions": "Confusing SaaS with managed services.",
    }

    props = build_propositions_for_chunk(mock_chunk)
    assert len(props) > 0, "No propositions generated"
    assert all("vendor lock-in" in p["text"].lower() for p in props)
    assert all(p["level"] in ("beginner", "mid", "senior") for p in props)

    # Test with ParentCache-like object
    class MockCache:
        def __init__(self):
            self.chunks = {"chunk_0000": mock_chunk}
    
    all_props = generate_all_propositions(MockCache())
    assert len(all_props) == len(props)
    assert all_props[0]["prop_id"] == "prop_00000"

    print(f"✅ proposition_generator tests passed — {len(props)} props for sample chunk")