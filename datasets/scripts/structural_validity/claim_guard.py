"""Vocabulary guard for sentences rendered from structural evidence.

Tree-sitter resolves no symbols. Every observable in this study is a name, a
path, or an interval comparison, so a rendered sentence must not dress one up as
data flow, control flow, a call graph, an alias, or def-use. The guard lives
apart from the measurement modules because it constrains what may be *written*
about a number, not how the number is computed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

FORBIDDEN_CLAIM_PHRASES: Final = (
    "shared variable",
    "data flow",
    "dataflow",
    "control flow",
    "controlflow",
    "call graph",
    "callgraph",
    "dependency entanglement",
    "alias",
    "def-use",
    "def use",
)
_NEGATION: Final = re.compile(r"\b(no|not|never|without|cannot|does not|do not|excludes?|neither|nor)\b")
_SENTENCE: Final = re.compile(r"(?<=[.;:])\s+|\n")


@dataclass(frozen=True, slots=True)
class ForbiddenClaimError(Exception):
    """A rendered claim used vocabulary this study cannot support."""

    phrase: str
    text: str


def validate_claim(text: str) -> str:
    """Reject a claim that asserts a relationship this evidence cannot support.

    Only positive claims are rejected. A sentence is allowed to *disclaim* the
    vocabulary -- the manuscript has to be able to say that no data-flow
    analysis was performed -- so a negation in the same sentence clears it.
    """
    for sentence in _SENTENCE.split(text):
        lowered = sentence.lower()
        if _NEGATION.search(lowered):
            continue
        for phrase in FORBIDDEN_CLAIM_PHRASES:
            if phrase in lowered:
                raise ForbiddenClaimError(phrase, sentence.strip())
    return text


__all__ = ["FORBIDDEN_CLAIM_PHRASES", "ForbiddenClaimError", "validate_claim"]
