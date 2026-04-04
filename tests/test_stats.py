import pandas as pd
import numpy as np
from src.stats import compute_stats, assess_shape, compute_correlation


def test_assess_shape_symmetric():
    assert "symmetric" in assess_shape(0.1, 0.0)


def test_assess_shape_right_skewed():
    assert "right-skewed" in assess_shape(1.5, 0.0)


def test_compute_stats_numeric():
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    stats = compute_stats(df)
    assert len(stats) == 1
    assert stats[0].mean == 3.0
    assert stats[0].missing == 0


def test_compute_stats_missing():
    df = pd.DataFrame({"value": [1.0, None, 3.0, None, 5.0]})
    stats = compute_stats(df)
    assert stats[0].missing == 2
    assert stats[0].missing_pct == 40.0


def test_compute_stats_categorical():
    df = pd.DataFrame({"category": ["a", "b", "a", None]})
    stats = compute_stats(df)
    assert stats[0].mean is None
    assert stats[0].missing == 1


def test_correlation():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
        }
    )
    corr = compute_correlation(df)
    assert corr.loc["a", "b"] == 1.0
