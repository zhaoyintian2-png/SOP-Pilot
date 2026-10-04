# -*- coding: utf-8 -*-
"""
================================================================================
  safety.py — High-Risk Question Interception (Human-in-the-Loop)
================================================================================
Pre-retrieval safety scan over the user question. If any high-risk keyword
from :data:`config.HIGH_RISK_KEYWORDS` matches, the question is intercepted
and escalated to HR / Security without calling the LLM.

Questions that merely ask *how to report / policy for / process* are allowed
through even if they contain risk-related words (see
:data:`config.SAFETY_WHITELIST`).
================================================================================
"""

from __future__ import annotations

from config import HIGH_RISK_KEYWORDS, HIGH_RISK_REPLY, SAFETY_WHITELIST


def check_high_risk(question: str) -> str | None:
    """Scan ``question`` for high-risk keywords.

    Returns the interception reply string if the question should be blocked,
    or ``None`` if it may proceed to retrieval + generation.
    """
    q_lower = question.lower()

    # Allow meta-questions about reporting / policy / process even when they
    # mention risk-related terms.
    if any(term in q_lower for term in SAFETY_WHITELIST):
        return None

    for kw in HIGH_RISK_KEYWORDS:
        if kw.lower() in q_lower:
            return HIGH_RISK_REPLY
    return None
