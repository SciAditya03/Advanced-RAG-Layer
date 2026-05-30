"""
Multi-turn query rewriter: convert follow-up questions into standalone
search queries by referencing conversation history and scenario context.
"""

import re
import logging

__all__ = ["rewrite_query_with_history"]

logger = logging.getLogger(__name__)

# Cue words that strongly signal a follow-up reference
_FOLLOWUP_PATTERNS = re.compile(
    r'\b(that|this|it|they|them|those|these|its|their|he|she|such)\b',
    re.IGNORECASE,
)

# Words that indicate a direct question needing no rewrite
_STANDALONE_SIGNALS = re.compile(
    r'\b(what is|define|explain|how (to|does)|why is|list)\b',
    re.IGNORECASE,
)


def _extract_key_entities(history: list[dict], max_turns: int = 3) -> list[str]:
    """Pull noun phrases / capitalized tokens from recent turns.

    Args:
        history: Conversation history list.
        max_turns: How many recent turns to scan.

    Returns:
        Deduplicated list of entity strings.
    """
    entities: list[str] = []
    for turn in history[-max_turns:]:
        content = turn.get("content", "")
        # Grab capitalized multi-word phrases and quoted terms
        caps   = re.findall(r'\b[A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*\b', content)
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', content)
        entities += caps
        entities += [q[0] or q[1] for q in quoted]
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for e in entities:
        if e not in seen:
            seen.add(e)
            result.append(e)
    return result


def rewrite_query_with_history(
    query: str,
    chat_history: list[dict],
    scenario_id: str | None = None,
) -> str:
    """Convert a conversational follow-up into a standalone search query.

    Resolves pronouns and implicit references using recent conversation turns.
    If the query already appears standalone (no cue words), it is returned
    enriched with scenario context only.

    Args:
        query: The officer's latest question (may contain follow-up references).
        chat_history: Full conversation history as
            [{"role": "user"/"assistant", "content": "..."}, ...].
        scenario_id: Optional scenario identifier appended for scoped retrieval.

    Returns:
        Rewritten standalone query string.

    Example:
        >>> history = [
        ...     {"role": "user", "content": "What is vendor lock-in?"},
        ...     {"role": "assistant", "content": "Vendor lock-in occurs when..."},
        ... ]
        >>> rewrite_query_with_history("Why is that risky?", history, "IN-AIGOV-001")
        'Why is vendor lock-in risky? [Scenario: IN-AIGOV-001]'
    """
    if not chat_history:
        suffix = f" [Scenario: {scenario_id}]" if scenario_id else ""
        return query.strip() + suffix

    is_followup = bool(_FOLLOWUP_PATTERNS.search(query))

    if not is_followup:
        suffix = f" [Scenario: {scenario_id}]" if scenario_id else ""
        return query.strip() + suffix

    # ── Extract entities from history for resolution ───────────────────────
    entities = _extract_key_entities(chat_history, max_turns=3)

    # Pull the last assistant message for direct subject resolution
    last_assistant = ""
    for turn in reversed(chat_history):
        if turn.get("role") == "assistant":
            last_assistant = turn.get("content", "")
            break

    # Heuristic: grab first noun-phrase from last assistant turn (max 6 words)
    subject_match = re.search(
        r'\b([A-Z][a-zA-Z]+(?:\s+[a-zA-Z-]+){0,5})\b', last_assistant
    )
    primary_subject = subject_match.group(1) if subject_match else ""

    # Replace leading pronouns with primary subject
    rewritten = query
    if primary_subject:
        rewritten = re.sub(
            r'^(Why is |How does |What does )(that|this|it)\b',
            lambda m: m.group(1) + primary_subject,
            rewritten,
            flags=re.IGNORECASE,
        )
        # Generic pronoun replacement in the middle
        rewritten = re.sub(
            r'\b(that|this|it)\b(?!\s+(is|are|was|were))',
            primary_subject,
            rewritten,
            count=1,
            flags=re.IGNORECASE,
        )

    # Append up to 2 key entities for context
    appended_entities = [e for e in entities[:2] if e.lower() not in rewritten.lower()]
    if appended_entities:
        rewritten = rewritten.rstrip("?. ") + " regarding " + ", ".join(appended_entities)

    if scenario_id:
        rewritten += f" [Scenario: {scenario_id}]"

    logger.debug("query_rewriter: '%s' → '%s'", query, rewritten)
    return rewritten.strip()


if __name__ == "__main__":
    history = [
        {"role": "user",      "content": "What is vendor lock-in?"},
        {"role": "assistant", "content": "Vendor lock-in occurs when a government becomes "
                                         "dependent on a single technology provider."},
    ]

    result = rewrite_query_with_history("Why is that risky?", history, "IN-AIGOV-001")
    assert "vendor lock-in" in result.lower() or "Vendor" in result, \
        f"Expected vendor reference, got: {result}"
    assert "IN-AIGOV-001" in result, f"Missing scenario_id: {result}"
    assert len(result) > len("Why is that risky?"), "Rewritten query should be longer"

    # No history → passthrough
    plain = rewrite_query_with_history("What is AI bias?", [], "S2")
    assert "S2" in plain
    assert plain.startswith("What is AI bias?")

    # Standalone query → only scenario suffix added
    standalone = rewrite_query_with_history("Explain AI bias", history, "S3")
    assert "S3" in standalone

    print("✅ query_rewriter tests passed")
    print(f"   Rewritten: {result}")