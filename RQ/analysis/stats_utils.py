"""
Statistical utility functions for RQ analysis.

Provides:
- vargha_delaney_a: Effect size calculation for Hamming Loss
- wilcoxon_signed_rank: Paired non-parametric significance test
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import rankdata, wilcoxon


def vargha_delaney_a(model_a: ArrayLike, model_b: ArrayLike) -> float:
    """
    Compute Vargha-Delaney Â₁₂ effect size for Hamming Loss comparison.

    Measures the probability that model_a yields higher Hamming Loss than model_b.
    Since lower Hamming Loss is better, Â₁₂ < 0.5 indicates model_a performs better.

    Formula: Â₁₂ = (R₁/m - (m+1)/2) / n

    Where:
        - R₁: rank sum of model_a in combined ranking
        - m: number of samples from model_a
        - n: number of samples from model_b

    Reference:
        Vargha, A., & Delaney, H. D. (2000). A Critique and Improvement of
        the CL Common Language Effect Size Statistics of McGraw and Wong.
        Journal of Educational and Behavioral Statistics, 25(2), 101-132.

    Args:
        model_a: Hamming Loss values from first model
        model_b: Hamming Loss values from second model

    Returns:
        Â₁₂ value in range [0, 1]:
            - Â₁₂ = 0.5: No difference between models
            - Â₁₂ > 0.5: model_a has higher HL (worse performance)
            - Â₁₂ < 0.5: model_a has lower HL (better performance)

    Example:
        >>> model_a_hl = [0.1, 0.2, 0.15]  # Lower HL = better
        >>> model_b_hl = [0.3, 0.25, 0.4]  # Higher HL = worse
        >>> vargha_delaney_a(model_a_hl, model_b_hl)
        0.0  # model_a is consistently better
    """
    model_a = np.asarray(model_a)
    model_b = np.asarray(model_b)

    n_a = len(model_a)
    n_b = len(model_b)

    if n_a == 0 or n_b == 0:
        return 0.5

    # Combine and rank
    combined = np.concatenate([model_a, model_b])
    ranks = rankdata(combined, method='average')

    # R₁: rank sum of model_a
    r1 = np.sum(ranks[:n_a])

    # Â₁₂ formula
    return float((r1 / n_a - (n_a + 1) / 2) / n_b)


def wilcoxon_signed_rank(model_a: ArrayLike, model_b: ArrayLike) -> float:
    """
    Perform Wilcoxon signed-rank test for paired Hamming Loss samples.

    Non-parametric test for paired samples. Used because:
    1. Each commit is evaluated by both models (naturally paired)
    2. Hamming Loss is bounded [0, 1] and often non-normal
    3. Controls for per-commit difficulty variance

    Args:
        model_a: Hamming Loss values from first model
        model_b: Hamming Loss values from second model

    Returns:
        Two-sided p-value. Small p-value (< 0.05) indicates
        significant difference between models.

    Raises:
        ValueError: If sample sizes don't match

    Example:
        >>> model_a_hl = [0.1, 0.2, 0.15, 0.3]
        >>> model_b_hl = [0.2, 0.25, 0.2, 0.35]
        >>> p = wilcoxon_signed_rank(model_a_hl, model_b_hl)
        >>> print(f"p = {p:.3f}")
    """
    model_a = np.asarray(model_a)
    model_b = np.asarray(model_b)

    if len(model_a) != len(model_b):
        raise ValueError(f"Sample size mismatch: {len(model_a)} vs {len(model_b)}")

    if len(model_a) == 0:
        return 1.0

    try:
        result = wilcoxon(model_a, model_b, alternative='two-sided')
        return float(result.pvalue)
    except ValueError:
        # All differences are zero
        return 1.0
