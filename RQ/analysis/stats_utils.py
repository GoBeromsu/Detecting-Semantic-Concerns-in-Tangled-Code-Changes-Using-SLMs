"""
Statistical utility functions for RQ analysis.

Provides:
- vargha_delaney_a: Effect size calculation (Vargha-Delaney Â₁₂)
- wilcoxon_signed_rank: Paired non-parametric significance test
- detect_outliers_iqr: Outlier detection using IQR method
- compute_pairwise_stats: Wrapper that computes stats + formatted values
- mean_ci: Commit-level mean with a normal confidence interval

Statistical Test Choice (following Arcuri & Briand 2014):
- Wilcoxon Signed-Rank Test is used for pairwise model comparisons because:
  1. Same commits are evaluated by different models (paired design)
  2. Tests whether the differences are symmetric around zero
  3. Pairs naturally with Vargha-Delaney Â₁₂ effect size
  4. Makes no distributional assumptions about metric values
"""

import ast
import json

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import norm, rankdata, sem, wilcoxon

# Default threshold for outlier detection
OUTLIER_THRESHOLD_IQR = 1.5

# Columns that, together with the commit, identify one evaluated row
DEFAULT_PAIR_KEY = ("context_len", "with_message")


def commit_key(cell) -> str:
    """Normalize a `shas` cell into a key that is stable across both result pipelines.

    The GPT pipeline writes the Python repr of the sha list (single quotes) while the
    unsloth pipeline writes JSON (double quotes). Joining on the raw string would match
    zero rows between the two, which reads as "these models share no commits" rather than
    as an encoding mismatch.
    """
    value = cell
    for _ in range(4):
        if isinstance(value, list):
            return "|".join(map(str, value))
        if not isinstance(value, str):
            return str(cell)
        try:
            value = json.loads(value)
            continue
        except (ValueError, TypeError):
            pass
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return str(value)


def align_paired(frames: dict, key=DEFAULT_PAIR_KEY) -> dict:
    """Restrict every frame to the rows they all share, aligned row-for-row.

    The paired tests below read `.values` positionally. That was safe only while every
    model emitted the same commits, in the same order, for every sweep — an assumption that
    holds until a model refuses to answer. The LoRA arm rejects a handful of generations per
    sweep, which shifts every later row, so a positional pair can compare two different
    commits while still passing the equal-length check.

    Rows are keyed by commit plus `key` plus an occurrence index, so the repeated sweeps of
    one commit stay distinct. When no rows are missing this reproduces the positional
    pairing exactly, which is why it can be applied to the published results unchanged.
    """
    key = list(key)
    keyed = {}
    for name, df in frames.items():
        df = df.copy()
        df["_commit"] = df["shas"].map(commit_key)
        df["_repeat"] = df.groupby(["_commit"] + key).cumcount()
        keyed[name] = df.set_index(["_commit"] + key + ["_repeat"])

    common = None
    for df in keyed.values():
        common = df.index if common is None else common.intersection(df.index)

    # Keep the first frame's original row order rather than the intersection's. The tests
    # are order-invariant, but preserving it means a complete set of results comes back
    # untouched, so this can be applied to the published runs without moving their numbers.
    first = next(iter(keyed.values())).index
    common = first[first.isin(common)]

    return {
        name: df.loc[common].reset_index().drop(columns=["_commit", "_repeat"])
        for name, df in keyed.items()
    }


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
    Perform Wilcoxon signed-rank test for paired samples.

    Evaluates whether the performance difference between two models is due to chance.

    Design Rationale:
    1. Paired Comparison: We compare models on the SAME set of commits.
       This controls for commit difficulty, label counts, and complexity.
    2. Two-sided Test: The null hypothesis is that the difference is random (chance).
       We test if the observed difference is too extreme to be just a coincidence.

    Interpretation:
    - p-value < 0.05: The performance difference is NOT by chance (Significant).
    - p-value >= 0.05: The difference could be due to random chance (Not significant).

    Reference:
        Arcuri, A., & Briand, L. (2014). A Hitchhiker's Guide to Statistical
        Tests for Assessing Randomized Algorithms in Software Engineering.

    Args:
        model_a: Metric values from first model (must align with model_b by commit)
        model_b: Metric values from second model (must align with model_a by commit)

    Returns:
        Two-sided p-value. Small p-value indicates significance (not by chance).

    Example:
        >>> model_a_hl = [0.1, 0.2, 0.15, 0.3]  # HL for commits 1-4
        >>> model_b_hl = [0.2, 0.25, 0.2, 0.35]  # HL for same commits 1-4
        >>> p = wilcoxon_signed_rank(model_a_hl, model_b_hl)
        >>> print(f"p = {p:.3f}")
    """
    model_a = np.asarray(model_a)
    model_b = np.asarray(model_b)

    if len(model_a) == 0 or len(model_b) == 0:
        return 1.0

    if len(model_a) != len(model_b):
        raise ValueError("Paired test requires equal-length samples")

    # 1. Compute differences (Paired nature)
    # We analyze the "difference" per commit, not the raw values.
    differences = model_a - model_b

    # 2. Remove zero differences (Handling Ties)
    # Standard practice for Wilcoxon test:
    # If a pair has difference = 0 (models perform exactly the same),
    # it provides no information about "direction" of difference.
    # Therefore, these ties are excluded from the ranking process.
    non_zero_diff = differences[differences != 0]

    if len(non_zero_diff) == 0:
        # All differences are zero -> Models are identical
        # p-value = 1.0 (100% chance they are the same)
        return 1.0

    try:
        result = wilcoxon(non_zero_diff, alternative='two-sided')
        return float(result.pvalue)
    except ValueError:
        # Not enough data points
        return 1.0


def detect_outliers_iqr(
    data: pd.Series, threshold: float = OUTLIER_THRESHOLD_IQR
) -> tuple:
    """
    Detect outliers using the Interquartile Range (IQR) method.

    Outliers are defined as values below Q1 - threshold*IQR or
    above Q3 + threshold*IQR.

    Args:
        data: Pandas Series of values to check for outliers
        threshold: IQR multiplier for outlier bounds (default: 1.5)

    Returns:
        Tuple of (outlier_mask, outlier_info):
            - outlier_mask: Boolean Series where True indicates outlier
            - outlier_info: Dict with count, values, bounds

    Example:
        >>> data = pd.Series([1, 2, 3, 4, 100])
        >>> mask, info = detect_outliers_iqr(data)
        >>> print(f"Found {info['count']} outliers")
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    outlier_mask = (data < lower_bound) | (data > upper_bound)

    outlier_info = {
        "count": int(outlier_mask.sum()),
        "values": data[outlier_mask].tolist(),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
    }

    return outlier_mask, outlier_info

# Formatting constants
P_VALUE_THRESHOLD = 0.001
DECIMAL_PLACES = 3


def mean_ci(df: pd.DataFrame, metric: str = "hamming_loss", confidence: float = 0.95) -> dict:
    """Commit-level mean of `metric` with a normal confidence interval.

    Each `avg_result` file holds every repeated run of the same commit, so rows are clustered
    by commit rather than independent. The metric is first averaged within each commit
    (keyed by the normalized `shas`), and the interval is then computed over those per-commit
    means with `scipy.stats.norm.interval` and `scipy.stats.sem`.

    Returns:
        dict with mean, ci_low, ci_high, n_commits, n_rows
    """
    required = {"shas", metric}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if df.empty:
        raise ValueError("Cannot summarize an empty data frame")
    if df["shas"].isna().any():
        raise ValueError("shas must not contain missing values")
    metric_values = pd.to_numeric(df[metric], errors="coerce")
    if metric_values.isna().any() or not np.isfinite(metric_values).all():
        raise ValueError(f"{metric} must contain finite numeric values")
    per_commit = metric_values.groupby(df["shas"].map(commit_key)).mean()
    n = len(per_commit)
    mean = float(per_commit.mean())
    if n > 1:
        ci_low, ci_high = norm.interval(confidence, loc=mean, scale=sem(per_commit))
    else:
        ci_low = ci_high = mean
    return {
        "mean": mean,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_commits": int(n),
        "n_rows": int(len(df)),
    }


def compute_pairwise_stats(model_a: ArrayLike, model_b: ArrayLike, sig_threshold: float = 0.05) -> dict:
    """
    Compute pairwise statistics with LaTeX-formatted values.

    Uses Wilcoxon signed-rank test for significance and Vargha-Delaney Â₁₂ for effect size,
    following SE research best practices (Arcuri & Briand 2014).

    Args:
        model_a: Hamming Loss values from first model
        model_b: Hamming Loss values from second model
        sig_threshold: Significance threshold (default: 0.05)

    Returns:
        Dictionary containing:
            - p_value: LaTeX-formatted p-value (str)
            - effect_size: Formatted Vargha-Delaney Â₁₂ (str)
            - significant: Whether p < sig_threshold (bool)

    Example:
        >>> stats = compute_pairwise_stats(model_a_hl, model_b_hl)
        >>> print(stats["p_value"])  # '$<$ 0.001'
    """
    p_value = wilcoxon_signed_rank(model_a, model_b)
    effect_size = vargha_delaney_a(model_a, model_b)

    return {
        "p_value": f"$<$ {P_VALUE_THRESHOLD}" if p_value < P_VALUE_THRESHOLD else f"{p_value:.{DECIMAL_PLACES}f}",
        "effect_size": f"{effect_size:.{DECIMAL_PLACES}f}",
        "significant": bool(p_value < sig_threshold)
    }
