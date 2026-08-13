"""Structural validity measurement and analysis.

Core modules for colocation analysis (same-file/same-folder pairing metrics).
The kept lane: colocation pipeline and diff-metrics foundation.
"""

from .claim_guard import FORBIDDEN_CLAIM_PHRASES, ForbiddenClaimError, validate_claim
from .colocation_cli import main as cli_main
from .colocation_data import PairRow, SummaryRow, summarize_by_split_k
from .colocation_report import build_summary_markdown
from .diff_metrics import pair_diff_metrics

__all__ = (
    "FORBIDDEN_CLAIM_PHRASES",
    "ForbiddenClaimError",
    "PairRow",
    "SummaryRow",
    "build_summary_markdown",
    "cli_main",
    "pair_diff_metrics",
    "summarize_by_split_k",
    "validate_claim",
)
