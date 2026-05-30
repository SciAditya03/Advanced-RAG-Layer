"""
Low-level string and dict utilities shared across the pipeline.
No imports from other pipeline modules — zero dependency leaf.
"""

import re
import logging

__all__ = ["_clean", "safe_get", "tokenize_for_bm25"]

logger = logging.getLogger(__name__)


def _clean(text: object) -> str:
    """Collapse whitespace and strip leading/trailing space.

    Args:
        text: Any object; will be cast to str.

    Returns:
        Cleaned string.
    """
    return re.sub(r'\s+', ' ', str(text or "")).strip()


def safe_get(d: dict, *keys: str, default: str = "") -> object:
    """Safely chain key access on nested dicts.

    Args:
        d: Root dictionary.
        *keys: Sequence of keys to traverse.
        default: Value returned when any key is missing or None.

    Returns:
        Value at the nested key path, or *default*.

    Example:
        >>> safe_get({"a": {"b": 1}}, "a", "b")
        1
        >>> safe_get({"a": None}, "a", "b", default="?")
        '?'
    """
    val = d
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k, default)
    return val if val is not None else default


def tokenize_for_bm25(text: str) -> list[str]:
    """Lowercase, remove punctuation, split on whitespace for BM25.

    Args:
        text: Raw string.

    Returns:
        List of lowercase tokens.

    Example:
        >>> tokenize_for_bm25("Hello, World!")
        ['hello', 'world']
    """
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return text.split()


if __name__ == "__main__":
    assert _clean("  hello   world  ") == "hello world"
    assert safe_get({"a": {"b": 42}}, "a", "b") == 42
    assert safe_get({"a": None}, "a", "b", default="x") == "x"
    assert tokenize_for_bm25("Hello, World!") == ["hello", "world"]
    print("✅ helpers tests passed")