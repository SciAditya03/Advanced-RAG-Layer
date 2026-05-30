"""
Map onboarding questionnaire responses to a structured officer profile.
"""

import logging

__all__ = ["questionnaire_to_profile"]

logger = logging.getLogger(__name__)

_ROLE_LEVEL_MAP = {
    "junior officer":    "beginner",
    "mid-level manager": "mid",
    "senior official":   "senior",
}

_COMFORT_LEVEL_MAP = {
    "not familiar":      "beginner",
    "somewhat familiar": "mid",
    "very familiar":     "senior",
}

_LEVEL_ORDER = ["beginner", "mid", "senior"]


def questionnaire_to_profile(responses: dict) -> dict:
    """Map onboarding questionnaire answers to an officer profile dict.

    Blends role-seniority and self-reported AI comfort, conservatively
    picking the lower of the two to avoid over-estimating capability.

    Args:
        responses: Dict with any of:
            - role       : 'junior officer' | 'mid-level manager' | 'senior official'
            - comfort    : 'not familiar' | 'somewhat familiar' | 'very familiar'
            - domain     : 'procurement' | 'data privacy' | 'policy design' | ...
            - work_shape : 'mostly operational' | 'mostly policy' | 'mixed'
            - name       : officer's preferred name (optional)

    Returns:
        Dict with keys: level, domain, orientation, preferred_name.

    Example:
        >>> questionnaire_to_profile({"role": "junior officer", "comfort": "not familiar"})
        {'level': 'beginner', 'domain': 'general', 'orientation': 'policy', 'preferred_name': 'Officer'}
    """
    role_level    = _ROLE_LEVEL_MAP.get(responses.get("role", "").lower(), "mid")
    comfort_level = _COMFORT_LEVEL_MAP.get(responses.get("comfort", "").lower(), role_level)

    # Conservative blend: take the lower level
    level = _LEVEL_ORDER[
        min(_LEVEL_ORDER.index(role_level), _LEVEL_ORDER.index(comfort_level))
    ]

    work_shape  = responses.get("work_shape", "")
    orientation = "operational" if "operational" in work_shape.lower() else "policy"

    profile = {
        "level":          level,
        "domain":         responses.get("domain", "general"),
        "orientation":    orientation,
        "preferred_name": responses.get("name", "Officer"),
    }

    logger.debug("questionnaire_to_profile: %s → %s", responses, profile)
    return profile


if __name__ == "__main__":
    r = questionnaire_to_profile({
        "role": "junior officer", "comfort": "not familiar",
        "domain": "procurement", "work_shape": "mostly operational", "name": "Priya",
    })
    assert r["level"] == "beginner"
    assert r["orientation"] == "operational"
    assert r["preferred_name"] == "Priya"

    # senior role + low comfort → beginner
    r2 = questionnaire_to_profile({"role": "senior official", "comfort": "not familiar"})
    assert r2["level"] == "beginner"

    print("✅ profile_mapper tests passed")