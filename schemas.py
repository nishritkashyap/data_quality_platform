"""
Pydantic v2 schemas for structured profiling output.

Every profiling result is a typed, serializable data model.
This is the contract between the data engine and all consumers
(API endpoints, CLI tools, report generators, frontend UI).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ColumnDataType(str, Enum):
    """Detected semantic data type for a column."""
    NUMERIC_INTEGER = "numeric_integer"
    NUMERIC_FLOAT = "numeric_float"
    TEXT = "text"
    BOOLEAN = "boolean"
    DATE = "date"
    EMAIL = "email"
    PHONE = "phone"
    CATEGORICAL = "categorical"
    CONSTANT = "constant"
    EMPTY = "empty"
    MIXED = "mixed"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class NumericStats(BaseModel):
    """Statistical summary for a numeric column."""
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    iqr: Optional[float] = None
    skewness: Optional[float] = None
    zero_count: int = 0
    negative_count: int = 0


class CategoricalStats(BaseModel):
    """Distribution summary for a categorical or text column."""
    top_values: dict[str, int] = Field(
        default_factory=dict,
        description="Top N value → count mapping",
    )
    unique_count: int = 0
    most_frequent_value: Optional[str] = None
    most_frequent_count: int = 0
    least_frequent_value: Optional[str] = None
    least_frequent_count: int = 0


class FormatPattern(BaseModel):
    """A regex pattern detected within a column (e.g. date, email, phone)."""
    pattern_name: str
    regex: str
    match_count: int = 0
    match_percentage: float = 0.0
    sample_matches: list[str] = Field(default_factory=list)
    sample_non_matches: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Column-level profile
# ---------------------------------------------------------------------------

class ColumnProfile(BaseModel):
    """Complete profile for a single column."""

    # Identity
    column_name: str
    column_index: int
    detected_type: ColumnDataType = ColumnDataType.UNKNOWN
    original_dtype: str = ""

    # Completeness
    total_values: int = 0
    missing_count: int = 0
    missing_percentage: float = 0.0
    non_null_count: int = 0

    # Uniqueness
    unique_count: int = 0
    unique_percentage: float = 0.0
    is_potential_id: bool = False

    # Numeric statistics (populated only for numeric columns)
    numeric_stats: Optional[NumericStats] = None

    # Categorical statistics (populated for text / categorical / boolean)
    categorical_stats: Optional[CategoricalStats] = None

    # Format patterns detected (date, email, phone regex matches)
    format_patterns: list[FormatPattern] = Field(default_factory=list)

    # Quality issues
    leading_trailing_whitespace_count: int = 0
    mixed_case_count: int = 0
    capitalization_styles: dict[str, int] = Field(
        default_factory=dict,
        description="Counts per style: UPPER, lower, Title, mIxEd",
    )

    # Sample values for quick inspection
    sample_values: list[Any] = Field(default_factory=list, max_length=10)


# ---------------------------------------------------------------------------
# Dataset-level profile
# ---------------------------------------------------------------------------

class DatasetProfile(BaseModel):
    """Complete profile for an entire dataset."""

    # Metadata
    filename: str = ""
    file_size_bytes: int = 0
    profiled_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Dimensions
    total_rows: int = 0
    total_columns: int = 0

    # Duplicate analysis
    exact_duplicate_rows: int = 0
    exact_duplicate_percentage: float = 0.0

    # Overall completeness
    total_missing_cells: int = 0
    total_cells: int = 0
    overall_completeness_percentage: float = 0.0

    # Problematic columns
    empty_columns: list[str] = Field(default_factory=list)
    constant_columns: list[str] = Field(default_factory=list)

    # Per-column profiles
    columns: list[ColumnProfile] = Field(default_factory=list)

    # Column name issues
    duplicate_column_names: list[str] = Field(default_factory=list)
    columns_with_special_chars: list[str] = Field(default_factory=list)

    # Type summary counts
    numeric_column_count: int = 0
    text_column_count: int = 0
    date_column_count: int = 0
    email_column_count: int = 0
    phone_column_count: int = 0
    boolean_column_count: int = 0
