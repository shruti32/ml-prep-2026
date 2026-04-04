import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Optional


@dataclass
class ColumnStats:
    name: str
    dtype: str
    count: int
    missing: int
    missing_pct: float
    mean: Optional[float]
    median: Optional[float]
    std: Optional[float]
    skewness: Optional[float]
    kurtosis: Optional[float]
    min: Optional[float]
    max: Optional[float]
    shape_assessment: Optional[str]


def assess_shape(skew: float, kurt: float) -> str:
    """Assess distribution shape based on skewness and kurtosis."""
    parts = []

    if abs(skew) < 0.5:
        parts.append("approximately symmetric")
    elif skew > 1:
        parts.append("highly right-skewed")
    elif skew > 0.5:
        parts.append("moderately right-skewed")
    elif skew < -1:
        parts.append("highly left-skewed")
    else:
        parts.append("moderately left-skewed")

    if kurt > 1:
        parts.append("heavy-tailed (leptokurtic)")
    elif kurt < -1:
        parts.append("thin-tailed (platykurtic)")
    else:
        parts.append("normal-tailed (mesokurtic)")

    return ", ".join(parts)


def compute_stats(df: pd.DataFrame) -> list[ColumnStats]:
    """Compute descriptive statistics for all columns in a DataFrame."""
    results = []

    for col in df.columns:
        series = df[col]
        missing = series.isna().sum()
        missing_pct = (missing / len(series)) * 100

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            skew = float(stats.skew(clean))
            kurt = float(stats.kurtosis(clean))

            results.append(
                ColumnStats(
                    name=col,
                    dtype=str(series.dtype),
                    count=len(clean),
                    missing=int(missing),
                    missing_pct=round(missing_pct, 2),
                    mean=round(float(clean.mean()), 4),
                    median=round(float(clean.median()), 4),
                    std=round(float(clean.std(ddof=1)), 4),
                    skewness=round(skew, 4),
                    kurtosis=round(kurt, 4),
                    min=round(float(clean.min()), 4),
                    max=round(float(clean.max()), 4),
                    shape_assessment=assess_shape(skew, kurt),
                )
            )
        else:
            results.append(
                ColumnStats(
                    name=col,
                    dtype=str(series.dtype),
                    count=int(series.notna().sum()),
                    missing=int(missing),
                    missing_pct=round(missing_pct, 2),
                    mean=None,
                    median=None,
                    std=None,
                    skewness=None,
                    kurtosis=None,
                    min=None,
                    max=None,
                    shape_assessment=None,
                )
            )

    return results


def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix for numeric columns only."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr().round(3)
