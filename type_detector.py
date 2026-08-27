"""
Type Detector - Infers the semantic data type of each column.

Detection order (most specific → least specific):
  1. Email   (regex)
  2. Date    (regex, multiple format patterns)
  3. Phone   (regex, with numeric exclusion guard)
  4. Boolean (value set membership)
  5. Numeric (integer vs float via pd.to_numeric coercion)
  6. Categorical vs free text (cardinality ratio heuristic)

Date is checked before phone because the phone regex is intentionally
broad and would otherwise match date strings like "2024-01-15".
"""

from __future__ import annotations

import re
from typing import Union

import pandas as pd

from data_engine.schemas import ColumnDataType, FormatPattern


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)

PHONE_RE = re.compile(r"^[\+]?[\d\s\-\.\(\)]{7,20}$")

INTEGER_RE = re.compile(r"^-?\d+$")

DATE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("YYYY-MM-DD",    re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$")),
    ("MM/DD/YYYY",    re.compile(r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}$")),
    ("DD.MM.YYYY",    re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4}$")),
    ("MM/DD/YY",      re.compile(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2}$")),
    ("Month DD, YYYY", re.compile(r"^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}$")),
    ("DD Month YYYY", re.compile(r"^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}$")),
]

BOOLEAN_VALUES: set[str] = {
    "true", "false", "yes", "no", "y", "n",
    "1", "0", "t", "f", "on", "off",
}

# ---------------------------------------------------------------------------
# Detection thresholds (centralised for easy tuning)
# ---------------------------------------------------------------------------

THRESHOLD_EMAIL: float = 70.0
THRESHOLD_DATE: float = 60.0
THRESHOLD_PHONE: float = 70.0
THRESHOLD_BOOLEAN: float = 80.0
THRESHOLD_NUMERIC: float = 80.0
THRESHOLD_CATEGORICAL_RATIO: float = 0.3
THRESHOLD_CATEGORICAL_MAX_UNIQUE: int = 20
THRESHOLD_FREE_TEXT_AVG_LEN: int = 50
MAX_SAMPLE_SIZE: int = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_non_null(series: pd.Series, max_sample: int = MAX_SAMPLE_SIZE) -> pd.Series:
    """Return a sample of non-null, non-empty string values."""
    clean = series.dropna().astype(str).str.strip()
    clean = clean[clean != ""]
    if len(clean) > max_sample:
        clean = clean.sample(n=max_sample, random_state=42)
    return clean


def _match_ratio(series: pd.Series, pattern: re.Pattern) -> tuple[pd.Series, float]:
    """Return (boolean mask, match percentage) for a regex against a series."""
    if len(series) == 0:
        return pd.Series(dtype=bool), 0.0
    mask = series.apply(lambda x: bool(pattern.match(str(x))))
    pct = float(mask.sum() / len(series) * 100)
    return mask, pct


def _build_format_pattern(
    name: str,
    regex: re.Pattern,
    sample: pd.Series,
    mask: pd.Series,
    pct: float,
) -> FormatPattern:
    """Build a FormatPattern from match results."""
    return FormatPattern(
        pattern_name=name,
        regex=regex.pattern,
        match_count=int(mask.sum()),
        match_percentage=round(pct, 1),
        sample_matches=sample[mask].head(5).tolist(),
        sample_non_matches=sample[~mask].head(5).tolist(),
    )


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def detect_column_type(
    series: pd.Series,
) -> tuple[ColumnDataType, list[FormatPattern]]:
    """
    Detect the semantic data type of a column.

    Args:
        series: A pandas Series (one column of data).

    Returns:
        Tuple of (detected type, list of format patterns found).
    """
    sample = _sample_non_null(series)
    patterns: list[FormatPattern] = []

    if len(sample) == 0:
        return ColumnDataType.EMPTY, []

    if sample.nunique() == 1:
        return ColumnDataType.CONSTANT, []

    total = len(sample)

    # 1. Email ─────────────────────────────────────────────────────────────
    email_mask, email_pct = _match_ratio(sample, EMAIL_RE)
    if email_pct >= THRESHOLD_EMAIL:
        patterns.append(_build_format_pattern("email", EMAIL_RE, sample, email_mask, email_pct))
        return ColumnDataType.EMAIL, patterns

    # 2. Date (checked before phone — phone regex is broad) ────────────────
    best_pct, best_name, best_re = 0.0, "", None
    for pat_name, pat_re in DATE_PATTERNS:
        _, pct = _match_ratio(sample, pat_re)
        if pct > best_pct:
            best_pct, best_name, best_re = pct, pat_name, pat_re

    if best_pct >= THRESHOLD_DATE and best_re is not None:
        date_mask, _ = _match_ratio(sample, best_re)
        patterns.append(_build_format_pattern(f"date_{best_name}", best_re, sample, date_mask, best_pct))
        return ColumnDataType.DATE, patterns

    # 3. Phone (with numeric exclusion guard) ──────────────────────────────
    phone_mask, phone_pct = _match_ratio(sample, PHONE_RE)
    _, int_pct = _match_ratio(sample, INTEGER_RE)
    numeric_coerced = pd.to_numeric(
        sample.astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    numeric_pct = float(numeric_coerced.notna().sum() / total * 100)

    if phone_pct >= THRESHOLD_PHONE and int_pct < 70 and numeric_pct < 70:
        patterns.append(_build_format_pattern("phone", PHONE_RE, sample, phone_mask, phone_pct))
        return ColumnDataType.PHONE, patterns

    # 4. Boolean ───────────────────────────────────────────────────────────
    bool_mask = sample.apply(lambda x: str(x).strip().lower() in BOOLEAN_VALUES)
    bool_pct = float(bool_mask.sum() / total * 100)
    if bool_pct >= THRESHOLD_BOOLEAN:
        return ColumnDataType.BOOLEAN, []

    # 5. Numeric (reuse coerced series from phone guard) ───────────────────
    if numeric_pct >= THRESHOLD_NUMERIC:
        valid_nums = numeric_coerced.dropna()
        if len(valid_nums) > 0 and (valid_nums == valid_nums.astype(int)).all():
            return ColumnDataType.NUMERIC_INTEGER, []
        return ColumnDataType.NUMERIC_FLOAT, []

    # 6. Categorical vs free text ──────────────────────────────────────────
    unique_ratio = sample.nunique() / total if total > 0 else 1.0

    if unique_ratio < THRESHOLD_CATEGORICAL_RATIO or sample.nunique() <= THRESHOLD_CATEGORICAL_MAX_UNIQUE:
        return ColumnDataType.CATEGORICAL, patterns

    avg_len = sample.str.len().mean()
    if avg_len > THRESHOLD_FREE_TEXT_AVG_LEN:
        return ColumnDataType.TEXT, patterns

    # Mixed signals (partially numeric, partially text)
    if numeric_pct >= 40:
        return ColumnDataType.MIXED, patterns

    return ColumnDataType.TEXT, patterns
