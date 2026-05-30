"""
Teaching-style prompt builder for the AI Governance Simulator.
Produces level-appropriate prompts grounded in retrieved context.
"""

import logging

__all__ = ["build_teaching_prompt"]

logger = logging.getLogger(__name__)

from ..config import STYLE_GUIDE

_PROMPT_TEMPLATE = """\
You are an AI governance instructor teaching a {level}-level officer in the Indian public sector.

TEACHING STYLE: {style}

RULES:
1. Use ONLY the context provided below. Do not hallucinate or add external knowledge.
2. Cite the source for every claim using this format: [Scenario: X | Topic: Y]
3. If multiple laws or principles are mentioned, note each explicitly.
4. End your response with exactly one reflection question appropriate for a {level}-level officer.

CONTEXT:
{context}

OFFICER QUERY: {query}

RESPONSE:"""


def build_teaching_prompt(context: str, query: str, officer_profile: dict) -> str:
    """Build a teaching-style prompt grounded in retrieved context.

    Args:
        context: Compressed context string from compress_for_officer.
        query: The officer's original question.
        officer_profile: Dict with at least 'level'; optionally 'preferred_name'.

    Returns:
        Formatted prompt string ready for LLM generation.

    Example:
        >>> prompt = build_teaching_prompt("some context", "What is AI bias?", {"level": "mid"})
        >>> "OFFICER QUERY" in prompt
        True
    """
    level = officer_profile.get("level", "mid")
    style = STYLE_GUIDE.get(level, STYLE_GUIDE["mid"])

    prompt = _PROMPT_TEMPLATE.format(
        level=level,
        style=style,
        context=context,
        query=query,
    )

    logger.debug("build_teaching_prompt: level=%s context_chars=%d", level, len(context))
    return prompt


if __name__ == "__main__":
    ctx    = "[Scenario: S1 | Topic: vendor lock-in]\nVendor lock-in is dependency."
    query  = "What is vendor lock-in?"

    for lvl in ("beginner", "mid", "senior"):
        prompt = build_teaching_prompt(ctx, query, {"level": lvl})
        assert "OFFICER QUERY" in prompt
        assert query in prompt
        assert ctx in prompt
        assert STYLE_GUIDE[lvl][:20] in prompt

    print("✅ prompt_builder tests passed")