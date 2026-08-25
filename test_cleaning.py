"""
Test suite for the Deterministic Cleaning Engine (Phase 2).

Covers:
  - Individual cleaners (whitespace, column names, capitalization,
    dates, emails, phones, empty/constant columns, missing values, duplicates)
  - CleaningPipeline (default config, custom config, audit trail)
  - Original DataFrame is never modified
  - Edge cases
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_engine.cleaning.cleaners import (
    fill_missing_values,
    normalize_capitalization,
    normalize_column_names,
    normalize_emails,
    normalize_phones,
    remove_constant_columns,
    remove_empty_columns,
    remove_exact_duplicates,
    standardize_dates,
    trim_whitespace,
)
from data_engine.cleaning.pipeline import CleaningConfig, CleaningPipeline
from tests.generate_synthetic_data import generate_messy_customer_data


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def messy_df() -> pd.DataFrame:
    return generate_messy_customer_data(num_rows=100, seed=42)


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """Small handcrafted DataFrame for precise assertions."""
    return pd.DataFrame({
        "Name": ["  Alice ", " BOB", "charlie  ", "  Diana  "],
        "Email": ["ALICE@TEST.COM", "bob@test.com", "invalid", "Diana@Test.Com"],
        "Phone": ["(555) 123-4567", "555.123.4567", "+15551234567", "12345"],
        "Date": ["01/15/2024", "2024-01-15", "15.01.2024", "bad-date"],
        "Revenue": [1000.0, 2000.0, None, 4000.0],
        "Empty Col": [None, None, None, None],
        "Status": ["Active", "Active", "Active", "Active"],
    })


# ═══════════════════════════════════════════════════════════════════════════
# Individual Cleaners
# ═══════════════════════════════════════════════════════════════════════════

class TestTrimWhitespace:

    def test_strips_whitespace(self, simple_df: pd.DataFrame):
        cleaned, result = trim_whitespace(simple_df)
        assert cleaned["Name"].iloc[0] == "Alice"
        assert cleaned["Name"].iloc[1] == "BOB"
        assert cleaned["Name"].iloc[2] == "charlie"
        assert result.has_changes

    def test_original_unchanged(self, simple_df: pd.DataFrame):
        original_copy = simple_df.copy()
        trim_whitespace(simple_df)
        pd.testing.assert_frame_equal(simple_df, original_copy)

    def test_no_changes_when_clean(self):
        df = pd.DataFrame({"a": ["clean", "values"]})
        _, result = trim_whitespace(df)
        assert not result.has_changes


class TestNormalizeColumnNames:

    def test_basic_normalization(self):
        df = pd.DataFrame({"First Name": [1], "Last-Name": [2], "Email Address!": [3]})
        cleaned, result = normalize_column_names(df)
        assert "first_name" in cleaned.columns
        assert "last_name" in cleaned.columns
        assert "email_address" in cleaned.columns
        assert result.has_changes

    def test_already_clean(self):
        df = pd.DataFrame({"first_name": [1], "email": [2]})
        _, result = normalize_column_names(df)
        assert not result.has_changes

    def test_special_characters_removed(self):
        df = pd.DataFrame({"col (a)": [1], "col$b": [2]})
        cleaned, _ = normalize_column_names(df)
        assert "col_a" in cleaned.columns
        assert "colb" in cleaned.columns


class TestNormalizeCapitalization:

    def test_title_case(self, simple_df: pd.DataFrame):
        cleaned, result = normalize_capitalization(
            simple_df, columns=["Name"], style="title",
        )
        assert cleaned["Name"].iloc[0] == "  Alice "  # whitespace untouched if not trimmed first
        # After applying title: "  Alice " stays same since it's already title-ish after apply

    def test_lower_case(self):
        df = pd.DataFrame({"city": ["NEW YORK", "Los Angeles", "CHICAGO"]})
        cleaned, result = normalize_capitalization(df, columns=["city"], style="lower")
        assert cleaned["city"].iloc[0] == "new york"
        assert cleaned["city"].iloc[2] == "chicago"

    def test_preserves_nulls(self):
        df = pd.DataFrame({"a": ["HELLO", None, "WORLD"]})
        cleaned, _ = normalize_capitalization(df, style="lower")
        assert pd.isna(cleaned["a"].iloc[1])


class TestStandardizeDates:

    def test_mixed_formats(self, simple_df: pd.DataFrame):
        cleaned, result = standardize_dates(simple_df, columns=["Date"])
        # "01/15/2024" and "2024-01-15" should both become "2024-01-15"
        assert cleaned["Date"].iloc[0] == "2024-01-15"
        assert cleaned["Date"].iloc[1] == "2024-01-15"
        assert result.has_changes

    def test_invalid_dates_preserved(self, simple_df: pd.DataFrame):
        cleaned, _ = standardize_dates(simple_df, columns=["Date"])
        # "bad-date" should be left unchanged
        assert cleaned["Date"].iloc[3] == "bad-date"

    def test_no_date_columns_skipped(self):
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        _, result = standardize_dates(df, columns=["name"])
        # Names won't parse as dates
        assert not result.has_changes


class TestNormalizeEmails:

    def test_lowercases_valid_emails(self, simple_df: pd.DataFrame):
        cleaned, result = normalize_emails(simple_df, columns=["Email"])
        assert cleaned["Email"].iloc[0] == "alice@test.com"
        assert cleaned["Email"].iloc[3] == "diana@test.com"
        assert result.has_changes

    def test_invalid_emails_unchanged(self, simple_df: pd.DataFrame):
        cleaned, _ = normalize_emails(simple_df, columns=["Email"])
        assert cleaned["Email"].iloc[2] == "invalid"


class TestNormalizePhones:

    def test_reformats_valid_phones(self, simple_df: pd.DataFrame):
        cleaned, result = normalize_phones(simple_df, columns=["Phone"])
        assert cleaned["Phone"].iloc[0] == "(555) 123-4567"
        assert cleaned["Phone"].iloc[1] == "(555) 123-4567"
        assert cleaned["Phone"].iloc[2] == "(555) 123-4567"  # +1 stripped

    def test_invalid_phones_unchanged(self, simple_df: pd.DataFrame):
        cleaned, _ = normalize_phones(simple_df, columns=["Phone"])
        assert cleaned["Phone"].iloc[3] == "12345"  # too short, left alone


class TestRemoveEmptyColumns:

    def test_removes_empty(self, simple_df: pd.DataFrame):
        cleaned, result = remove_empty_columns(simple_df)
        assert "Empty Col" not in cleaned.columns
        assert result.has_changes

    def test_keeps_partial_columns(self):
        df = pd.DataFrame({"a": [1, None], "b": [None, None]})
        cleaned, _ = remove_empty_columns(df)
        assert "a" in cleaned.columns
        assert "b" not in cleaned.columns


class TestRemoveConstantColumns:

    def test_removes_constant(self, simple_df: pd.DataFrame):
        cleaned, result = remove_constant_columns(simple_df)
        assert "Status" not in cleaned.columns
        assert result.has_changes

    def test_keeps_varied_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 1, 1]})
        cleaned, _ = remove_constant_columns(df)
        assert "a" in cleaned.columns
        assert "b" not in cleaned.columns


class TestFillMissingValues:

    def test_strategy_none(self):
        df = pd.DataFrame({"a": [1, None, 3]})
        _, result = fill_missing_values(df, strategy="none")
        assert not result.has_changes

    def test_strategy_value(self):
        df = pd.DataFrame({"a": ["x", None, "z"]})
        cleaned, result = fill_missing_values(df, strategy="value", fill_value="UNKNOWN")
        assert cleaned["a"].iloc[1] == "UNKNOWN"
        assert result.has_changes

    def test_strategy_mode(self):
        df = pd.DataFrame({"a": ["x", "x", None]})
        cleaned, result = fill_missing_values(df, strategy="mode")
        assert cleaned["a"].iloc[2] == "x"

    def test_column_selection(self):
        df = pd.DataFrame({"a": [None], "b": [None]})
        cleaned, result = fill_missing_values(
            df, strategy="value", fill_value="X", columns=["a"],
        )
        assert cleaned["a"].iloc[0] == "X"
        assert pd.isna(cleaned["b"].iloc[0])


class TestRemoveExactDuplicates:

    def test_removes_dupes(self):
        df = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
        cleaned, result = remove_exact_duplicates(df)
        assert len(cleaned) == 2
        assert result.has_changes
        assert result.actions[0].records_affected == 1

    def test_no_dupes(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        _, result = remove_exact_duplicates(df)
        assert not result.has_changes


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestPipeline:

    def test_default_config(self, messy_df: pd.DataFrame):
        pipeline = CleaningPipeline()
        output = pipeline.clean(messy_df)
        assert output.total_changes > 0
        assert output.rows_after <= output.rows_before
        assert "empty_column" not in output.cleaned_df.columns  # removed

    def test_original_not_modified(self, messy_df: pd.DataFrame):
        original_copy = messy_df.copy()
        pipeline = CleaningPipeline()
        pipeline.clean(messy_df)
        pd.testing.assert_frame_equal(messy_df, original_copy)

    def test_audit_trail(self, messy_df: pd.DataFrame):
        pipeline = CleaningPipeline()
        output = pipeline.clean(messy_df)
        assert output.audit_logger.entry_count > 0
        entries = output.audit_logger.get_entries()
        for entry in entries:
            assert entry.action != ""
            assert entry.reason != ""

    def test_custom_config(self, simple_df: pd.DataFrame):
        config = CleaningConfig(
            trim_whitespace=True,
            normalize_column_names=True,
            normalize_capitalization=True,
            capitalization_columns=["Name"],
            capitalization_style="title",
            normalize_emails=True,
            email_columns=["Email"],
            remove_empty_columns=True,
            remove_constant_columns=True,
            remove_exact_duplicates=False,
        )
        pipeline = CleaningPipeline(config)
        output = pipeline.clean(simple_df)
        assert output.total_changes > 0
        # Column names should be normalized
        assert "name" in output.cleaned_df.columns or "Name" not in output.cleaned_df.columns

    def test_all_off(self, simple_df: pd.DataFrame):
        config = CleaningConfig(
            trim_whitespace=False,
            normalize_column_names=False,
            normalize_emails=False,
            remove_empty_columns=False,
            remove_exact_duplicates=False,
        )
        pipeline = CleaningPipeline(config)
        output = pipeline.clean(simple_df)
        assert output.total_changes == 0

    def test_summary_output(self, messy_df: pd.DataFrame):
        pipeline = CleaningPipeline()
        output = pipeline.clean(messy_df)
        summary = output.summary()
        assert "Cleaning Summary" in summary
        assert "Rows:" in summary
        assert "Actions performed:" in summary
