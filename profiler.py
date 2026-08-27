"""
DataProfiler - The main profiling engine.

Analyses an entire dataset *without modifying it* and produces a
structured ``DatasetProfile`` containing:
  - Dimensions (rows, columns)
  - Duplicate row count
  - Per-column profiles (type, completeness, uniqueness, stats, patterns)
  - Empty and constant column detection
  - Column name issues
  - Overall completeness score
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Union

import pandas as pd

from data_engine.loader import get_file_metadata, load_dataframe
from data_engine.profiling.type_detector import detect_column_type
from data_engine.schemas import (
    CategoricalStats,
    ColumnDataType,
    ColumnProfile,
    DatasetProfile,
    NumericStats,
)

# Mapping from ColumnDataType → summary bucket name
_TYPE_BUCKET: dict[ColumnDataType, str] = {
    ColumnDataType.NUMERIC_INTEGER: "numeric",
    ColumnDataType.NUMERIC_FLOAT:   "numeric",
    ColumnDataType.TEXT:             "text",
    ColumnDataType.CATEGORICAL:     "text",
    ColumnDataType.MIXED:           "text",
    ColumnDataType.DATE:            "date",
    ColumnDataType.EMAIL:           "email",
    ColumnDataType.PHONE:           "phone",
    ColumnDataType.BOOLEAN:         "boolean",
}

_SPECIAL_CHAR_RE = re.compile(r"[^a-zA-Z0-9_ ]")


class DataProfiler:
    """
    Profile a dataset without modifying it.

    Usage::

        profiler = DataProfiler()
        profile = profiler.profile_file("data.csv")
        # or
        profile = profiler.profile_dataframe(df, filename="data.csv")
    """

    def __init__(self, top_n_categories: int = 10, sample_size: int = 10):
        self.top_n_categories = top_n_categories
        self.sample_size = sample_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile_file(self, filepath: Union[str, Path]) -> DatasetProfile:
        """Profile a CSV or XLSX file. The original file is never modified."""
        path = Path(filepath).resolve()
        metadata = get_file_metadata(path)
        df = load_dataframe(path)

        profile = self.profile_dataframe(df, filename=metadata["filename"])
        profile.file_size_bytes = metadata["size_bytes"]
        return profile

    def profile_dataframe(
        self,
        df: pd.DataFrame,
        filename: str = "",
    ) -> DatasetProfile:
        """
        Profile a Pandas DataFrame.

        Args:
            df:       The DataFrame to profile (will NOT be modified).
            filename: Optional filename for metadata.

        Returns:
            A complete ``DatasetProfile``.
        """
        profile = DatasetProfile(
            filename=filename,
            profiled_at=datetime.now(UTC),
            total_rows=len(df),
            total_columns=len(df.columns),
        )

        # Duplicate rows
        if profile.total_rows > 0:
            profile.exact_duplicate_rows = int(df.duplicated(keep="first").sum())
            profile.exact_duplicate_percentage = round(
                profile.exact_duplicate_rows / profile.total_rows * 100, 2
            )

        # Column name issues
        self._detect_column_name_issues(df, profile)

        # Per-column profiling
        total_missing = 0
        for idx, col_name in enumerate(df.columns):
            col_profile = self._profile_column(df[col_name], str(col_name), idx)
            profile.columns.append(col_profile)
            total_missing += col_profile.missing_count

        # Overall completeness
        profile.total_cells = profile.total_rows * profile.total_columns
        profile.total_missing_cells = total_missing
        if profile.total_cells > 0:
            profile.overall_completeness_percentage = round(
                (1 - total_missing / profile.total_cells) * 100, 2
            )

        # Empty / constant column lists
        profile.empty_columns = [
            c.column_name for c in profile.columns
            if c.detected_type == ColumnDataType.EMPTY
        ]
        profile.constant_columns = [
            c.column_name for c in profile.columns
            if c.detected_type == ColumnDataType.CONSTANT
        ]

        # Type summary counts
        type_counter: Counter[str] = Counter()
        for col in profile.columns:
            bucket = _TYPE_BUCKET.get(col.detected_type)
            if bucket:
                type_counter[bucket] += 1

        profile.numeric_column_count = type_counter["numeric"]
        profile.text_column_count = type_counter["text"]
        profile.date_column_count = type_counter["date"]
        profile.email_column_count = type_counter["email"]
        profile.phone_column_count = type_counter["phone"]
        profile.boolean_column_count = type_counter["boolean"]

        return profile

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_column_name_issues(df: pd.DataFrame, profile: DatasetProfile) -> None:
        """Check for duplicate column names and special characters."""
        col_names = [str(c) for c in df.columns]
        name_counts = Counter(col_names)

        profile.duplicate_column_names = [
            name for name, count in name_counts.items() if count > 1
        ]
        profile.columns_with_special_chars = [
            name for name in col_names if _SPECIAL_CHAR_RE.search(name)
        ]

    def _profile_column(
        self,
        series: pd.Series,
        col_name: str,
        col_index: int,
    ) -> ColumnProfile:
        """Profile a single column."""
        total = len(series)
        missing = int(series.isna().sum())
        non_null = series.dropna()
        unique = int(non_null.nunique())
        non_null_count = total - missing

        profile = ColumnProfile(
            column_name=col_name,
            column_index=col_index,
            original_dtype=str(series.dtype),
            total_values=total,
            missing_count=missing,
            non_null_count=non_null_count,
            missing_percentage=round(missing / total * 100, 2) if total > 0 else 0.0,
            unique_count=unique,
            unique_percentage=round(unique / non_null_count * 100, 2) if non_null_count > 0 else 0.0,
            is_potential_id=(unique == total and missing == 0 and total > 1),
        )

        # Type detection
        detected_type, format_patterns = detect_column_type(series)
        profile.detected_type = detected_type
        profile.format_patterns = format_patterns

        # Whitespace & capitalisation (string columns only)
        if pd.api.types.is_string_dtype(non_null) or non_null.dtype == object:
            self._analyse_string_quality(non_null, profile)

        # Numeric statistics
        if detected_type in (ColumnDataType.NUMERIC_INTEGER, ColumnDataType.NUMERIC_FLOAT):
            profile.numeric_stats = self._compute_numeric_stats(non_null)

        # Categorical statistics
        if detected_type in (
            ColumnDataType.CATEGORICAL, ColumnDataType.TEXT,
            ColumnDataType.MIXED, ColumnDataType.BOOLEAN,
        ):
            profile.categorical_stats = self._compute_categorical_stats(non_null)

        # Sample values
        profile.sample_values = non_null.head(self.sample_size).tolist()

        return profile

    @staticmethod
    def _analyse_string_quality(non_null: pd.Series, profile: ColumnProfile) -> None:
        """Detect whitespace issues and capitalisation inconsistencies."""
        str_vals = non_null.astype(str)

        # Whitespace
        has_ws = str_vals.str.len() != str_vals.str.strip().str.len()
        profile.leading_trailing_whitespace_count = int(has_ws.sum())

        # Capitalisation (vectorised)
        stripped = str_vals.str.strip()
        stripped = stripped[stripped != ""]
        if len(stripped) == 0:
            return

        is_upper = stripped.str.isupper()
        is_lower = stripped.str.islower()
        is_title = stripped.str.istitle()

        cap_styles = {
            "UPPER": int(is_upper.sum()),
            "lower": int(is_lower.sum()),
            "Title": int(is_title.sum()),
            "mIxEd": int((~is_upper & ~is_lower & ~is_title).sum()),
        }
        profile.capitalization_styles = cap_styles

        active_styles = sum(1 for v in cap_styles.values() if v > 0)
        if active_styles > 1:
            dominant_style = max(cap_styles, key=cap_styles.get)  # type: ignore[arg-type]
            profile.mixed_case_count = sum(
                c for style, c in cap_styles.items() if style != dominant_style
            )

    @staticmethod
    def _compute_numeric_stats(series: pd.Series) -> NumericStats:
        """Compute descriptive statistics for a numeric column."""
        numeric = pd.to_numeric(
            series.astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        ).dropna()

        if len(numeric) == 0:
            return NumericStats()

        q1 = float(numeric.quantile(0.25))
        q3 = float(numeric.quantile(0.75))

        stats = NumericStats(
            min=float(numeric.min()),
            max=float(numeric.max()),
            mean=round(float(numeric.mean()), 4),
            median=float(numeric.median()),
            std=round(float(numeric.std()), 4) if len(numeric) > 1 else 0.0,
            q1=q1,
            q3=q3,
            iqr=round(q3 - q1, 4),
            zero_count=int((numeric == 0).sum()),
            negative_count=int((numeric < 0).sum()),
        )

        if len(numeric) >= 3:
            try:
                stats.skewness = round(float(numeric.skew()), 4)
            except Exception:
                pass

        return stats

    def _compute_categorical_stats(self, series: pd.Series) -> CategoricalStats:
        """Compute frequency statistics for a categorical column."""
        counts = series.astype(str).value_counts()

        if len(counts) == 0:
            return CategoricalStats()

        return CategoricalStats(
            top_values=counts.head(self.top_n_categories).to_dict(),
            unique_count=len(counts),
            most_frequent_value=str(counts.index[0]),
            most_frequent_count=int(counts.iloc[0]),
            least_frequent_value=str(counts.index[-1]) if len(counts) > 1 else None,
            least_frequent_count=int(counts.iloc[-1]) if len(counts) > 1 else 0,
        )
