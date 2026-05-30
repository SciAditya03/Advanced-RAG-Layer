"""
Format GovernanceRAGPipeline output for Phase 6 Dynamic Section Learning.
"""

import logging

__all__ = ["format_for_teaching_section"]

logger = logging.getLogger(__name__)


def format_for_teaching_section(pipeline_output: dict) -> dict:
    """Convert GovernanceRAGPipeline.run() output for the simulator's Phase 6.

    Args:
        pipeline_output: Dict returned by GovernanceRAGPipeline.run().

    Returns:
        Dict with keys:
            - theory              : str  — full generated answer
            - citations           : list[dict] — [{scenario_id, topic, chunk_id}]
            - reflection_question : str  — extracted reflection prompt
            - verification_status : dict — CRAG verification log
            - source_chunks       : list[dict] — [{scenario_id, topic}]

    Example:
        >>> out = format_for_teaching_section({
        ...     "answer": "Answer text",
        ...     "citations": [{"scenario_id": "S1", "topic": "T1", "chunk_id": "c0"}],
        ...     "reflection_question": "Why does this matter?",
        ...     "verification_log": {"supported": True},
        ...     "retrieved_propositions": [{"scenario_id": "S1", "topic": "T1"}],
        ... })
        >>> out["theory"]
        'Answer text'
    """
    source_chunks = [
        {"scenario_id": h["scenario_id"], "topic": h["topic"]}
        for h in pipeline_output.get("retrieved_propositions", [])
    ]

    result = {
        "theory":               pipeline_output.get("answer", ""),
        "citations":            pipeline_output.get("citations", []),
        "reflection_question":  pipeline_output.get("reflection_question", ""),
        "verification_status":  pipeline_output.get("verification_log", {}),
        "source_chunks":        source_chunks,
    }

    logger.debug("format_for_teaching_section: %d citations, %d source chunks",
                 len(result["citations"]), len(result["source_chunks"]))
    return result


if __name__ == "__main__":
    mock_output = {
        "answer": "AI governance requires transparency.",
        "citations": [{"scenario_id": "IN-AIGOV-001", "topic": "vendor lock-in", "chunk_id": "chunk_0000"}],
        "reflection_question": "What risks concern you most?",
        "verification_log": {"supported": True, "unsupported_claims": []},
        "retrieved_propositions": [{"scenario_id": "IN-AIGOV-001", "topic": "vendor lock-in"}],
    }
    result = format_for_teaching_section(mock_output)
    assert result["theory"] == "AI governance requires transparency."
    assert result["source_chunks"][0]["scenario_id"] == "IN-AIGOV-001"
    print("✅ output_formatter tests passed")