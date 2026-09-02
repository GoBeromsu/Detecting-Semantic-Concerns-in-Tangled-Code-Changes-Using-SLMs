from unittest.mock import patch

import pandas as pd
import pytest

from RQ.analysis.stats_utils import mean_ci
from RQ.analysis.summary_ci import format_row


def sample_rows():
    return pd.DataFrame(
        {
            "shas": ["['a']"] * 3 + ['["b"]'] * 3,
            "hamming_loss": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        }
    )


def test_mean_ci_clusters_runs_and_uses_scipy_interval():
    with (
        patch("RQ.analysis.stats_utils.sem", return_value=0.25) as standard_error,
        patch("RQ.analysis.stats_utils.norm.interval", return_value=(0.1, 0.9)) as interval,
    ):
        result = mean_ci(sample_rows())

    standard_error.assert_called_once()
    assert standard_error.call_args.args[0].tolist() == [0.0, 1.0]
    interval.assert_called_once()
    assert interval.call_args.args[0] == 0.95
    assert interval.call_args.kwargs == {"loc": 0.5, "scale": 0.25}
    assert result == {
        "mean": 0.5,
        "ci_low": 0.1,
        "ci_high": 0.9,
        "n_commits": 2,
        "n_rows": 6,
    }


def test_format_row_formats_mean_and_interval():
    formatted = format_row(
        {
            "mean": 0.125,
            "ci_low": 0.075,
            "ci_high": 0.175,
            "n_commits": 8,
            "n_rows": 24,
        }
    )
    assert formatted["hl_ci"] == "0.12 [0.07, 0.17]"


@pytest.mark.parametrize(
    "rows, message",
    [
        (pd.DataFrame({"shas": ["a"]}), "Missing required columns"),
        (
            pd.DataFrame(columns=["shas", "hamming_loss"]),
            "empty data frame",
        ),
        (
            pd.DataFrame({"shas": [None], "hamming_loss": [0.0]}),
            "shas must not contain missing",
        ),
        (
            pd.DataFrame({"shas": ["a"], "hamming_loss": [float("inf")]}),
            "finite numeric values",
        ),
    ],
)
def test_mean_ci_rejects_invalid_input(rows, message):
    with pytest.raises(ValueError, match=message):
        mean_ci(rows)
