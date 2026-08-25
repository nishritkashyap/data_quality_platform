"""
Test suite for the Data Profiling Engine (Phase 1).

Covers:
  - File validation & loading
  - Column type detection (all 9 types)
  - Dataset profiling (dimensions, duplicates, missing, empty, constant,
    whitespace, capitalisation, numeric stats, serialization)
  - Edge cases (empty DF, single row, all nulls, special-char columns)
  - Audit logger (log, multi-entry, serialization, clear)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_engine.audit.logger import ActionType, AuditLogger
from data_engine.loader import DataLoadError, load_dataframe, validate_file
from data_engine.profiling.profiler import DataProfiler
from data_engine.profiling.type_detector import detect_column_type
from data_engine.schemas import ColumnDataType
from tests.generate_synthetic_data import generate_messy_customer_data


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def messy_df() -> pd.DataFrame:
    return generate_messy_customer_data(num_rows=200, seed=42)


@pytest.fixture
def profiler() -> DataProfiler:
    return DataProfiler()


@pytest.fixture
def tmp_csv(messy_df: pd.DataFrame, tmp_path: Path) -> Path:
    path = tmp_path / "messy.csv"
    messy_df.to_csv(path, index=False)
    return path


@pytest.fixture
def tmp_xlsx(messy_df: pd.DataFrame, tmp_path: Path) -> Path:
    path = tmp_path / "messy.xlsx"
    messy_df.to_excel(path, index=False)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# File Validation
# ═══════════════════════════════════════════════════════════════════════════

class TestFileValidation:

    def test_validate_csv(self, tmp_csv: Path):
        assert validate_file(tmp_csv).exists()

    def test_validate_xlsx(self, tmp_xlsx: Path):
        assert validate_file(tmp_xlsx).exists()

    def test_reject_nonexistent(self):
        with pytest.raises(DataLoadError, match="File not found"):
            validate_file("/nonexistent/file.csv")

    def test_reject_unsupported_extension(self, tmp_path: Path):
        bad = tmp_path / "data.txt"
        bad.write_text("hello")
        with pytest.raises(DataLoadError, match="Unsupported file type"):
            validate_file(bad)

    def test_load_csv(self, tmp_csv: Path):
        df = load_dataframe(tmp_csv)
        assert isinstance(df, pd.DataFrame) and len(df) > 0

    def test_load_xlsx(self, tmp_xlsx: Path):
        df = load_dataframe(tmp_xlsx)
        assert isinstance(df, pd.DataFrame) and len(df) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Type Detection
# ═══════════════════════════════════════════════════════════════════════════

class TestTypeDetection:

    def test_integer(self):
        dtype, _ = detect_column_type(pd.Series([str(i) for i in range(100)]))
        assert dtype == ColumnDataType.NUMERIC_INTEGER

    def test_float(self):
        dtype, _ = detect_column_type(pd.Series([f"{i}.{i}" for i in range(1, 50)]))
        assert dtype == ColumnDataType.NUMERIC_FLOAT

    def test_email(self):
        dtype, patterns = detect_column_type(
            pd.Series([f"user{i}@example.com" for i in range(50)])
        )
        assert dtype == ColumnDataType.EMAIL
        assert patterns[0].pattern_name == "email"

    def test_phone(self):
        dtype, patterns = detect_column_type(
            pd.Series([f"(555) 123-{i:04d}" for i in range(50)])
        )
        assert dtype == ColumnDataType.PHONE
        assert len(patterns) > 0

    def test_date(self):
        dtype, patterns = detect_column_type(
            pd.Series([f"2024-01-{d:02d}" for d in range(1, 29)])
        )
        assert dtype == ColumnDataType.DATE
        assert len(patterns) > 0

    def test_boolean(self):
        dtype, _ = detect_column_type(
            pd.Series(["yes", "no", "true", "false", "Y", "N"] * 10)
        )
        assert dtype == ColumnDataType.BOOLEAN

    def test_categorical(self):
        dtype, _ = detect_column_type(pd.Series(["A", "B", "C"] * 20))
        assert dtype == ColumnDataType.CATEGORICAL

    def test_empty(self):
        dtype, _ = detect_column_type(pd.Series([None, None, np.nan]))
        assert dtype == ColumnDataType.EMPTY

    def test_constant(self):
        dtype, _ = detect_column_type(pd.Series(["SAME"] * 50))
        assert dtype == ColumnDataType.CONSTANT


# ═══════════════════════════════════════════════════════════════════════════
# Profiler
# ═══════════════════════════════════════════════════════════════════════════

class TestProfiler:

    def test_dimensions(self, profiler: DataProfiler, messy_df: pd.DataFrame):
        p = profiler.profile_dataframe(messy_df, filename="test.csv")
        assert p.total_rows == len(messy_df)
        assert p.total_columns == len(messy_df.columns)
        assert p.filename == "test.csv"

    def test_duplicates(self, profiler: DataProfiler, messy_df: pd.DataFrame):
        p = profiler.profile_dataframe(messy_df)
        assert p.exact_duplicate_rows > 0
        assert p.exact_duplicate_percentage > 0

    def test_missing_values(self, profiler: DataProfiler, messy_df: pd.DataFrame):
        p = profiler.profile_dataframe(messy_df)
        assert p.total_missing_cells > 0
        assert p.overall_completeness_percentage < 100

    def test_empty_column(self, profiler: DataProfiler, messy_df: pd.DataFrame):
        p = profiler.profile_dataframe(messy_df)
        assert "empty_column" in p.empty_columns

    def test_constant_column(self, profiler: DataProfiler, messy_df: pd.DataFrame):
        p = profiler.profile_dataframe(messy_df)
        assert "constant_flag" in p.constant_columns

    def test_column_count(self, profiler: DataProfiler, messy_df: pd.DataFrame):
        p = profiler.profile_dataframe(messy_df)
        assert len(p.columns) == len(messy_df.columns)

    def test_numeric_stats(self, profiler: DataProfiler, messy_df: pd.DataFrame):
        p = profiler.profile_dataframe(messy_df)
        rev = next((c for c in p.columns if c.column_name == "annual_revenue"), None)
        if rev and rev.numeric_stats:
            s = rev.numeric_stats
            assert s.min is not None
            assert s.max is not None
            assert s.mean is not None
            assert s.q1 is not None and s.q3 is not None

    def test_whitespace(self, profiler: DataProfiler, messy_df: pd.DataFrame):
        p = profiler.profile_dataframe(messy_df)
        name_col = next(c for c in p.columns if c.column_name == "customer_name")
        assert name_col.leading_trailing_whitespace_count > 0

    def test_mixed_case(self, profiler: DataProfiler, messy_df: pd.DataFrame):
        p = profiler.profile_dataframe(messy_df)
        name_col = next(c for c in p.columns if c.column_name == "customer_name")
        active = sum(1 for v in name_col.capitalization_styles.values() if v > 0)
        assert active >= 2

    def test_sample_values(self, profiler: DataProfiler, messy_df: pd.DataFrame):
        p = profiler.profile_dataframe(messy_df)
        for col in p.columns:
            if col.detected_type != ColumnDataType.EMPTY:
                assert len(col.sample_values) > 0 or col.missing_count == col.total_values

    def test_profile_csv(self, profiler: DataProfiler, tmp_csv: Path):
        p = profiler.profile_file(tmp_csv)
        assert p.total_rows > 0 and p.file_size_bytes > 0

    def test_profile_xlsx(self, profiler: DataProfiler, tmp_xlsx: Path):
        p = profiler.profile_file(tmp_xlsx)
        assert p.total_rows > 0

    def test_json_serialization(self, profiler: DataProfiler, messy_df: pd.DataFrame):
        p = profiler.profile_dataframe(messy_df)
        json_str = p.model_dump_json(indent=2)
        assert '"total_rows"' in json_str and '"columns"' in json_str


# ═══════════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_empty_dataframe(self, profiler: DataProfiler):
        p = profiler.profile_dataframe(pd.DataFrame())
        assert p.total_rows == 0 and p.total_columns == 0

    def test_single_row(self, profiler: DataProfiler):
        p = profiler.profile_dataframe(pd.DataFrame({"a": [1], "b": ["x"]}))
        assert p.total_rows == 1 and p.total_columns == 2

    def test_all_nulls(self, profiler: DataProfiler):
        df = pd.DataFrame({"a": [None, None], "b": [np.nan, np.nan]})
        p = profiler.profile_dataframe(df)
        assert p.overall_completeness_percentage == 0.0

    def test_special_chars_in_column_names(self, profiler: DataProfiler):
        df = pd.DataFrame({"col (a)": [1, 2], "col-b!": [3, 4]})
        p = profiler.profile_dataframe(df)
        assert len(p.columns_with_special_chars) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Audit Logger
# ═══════════════════════════════════════════════════════════════════════════

class TestAuditLogger:

    def test_log_entry(self):
        logger = AuditLogger()
        entry = logger.log(
            action_type=ActionType.PROFILING,
            action="profile_dataset",
            reason="Profiling completed",
            details={"rows": 100},
        )
        assert entry.action == "profile_dataset"
        assert logger.entry_count == 1

    def test_multiple_entries(self):
        logger = AuditLogger()
        logger.log(action_type=ActionType.PROFILING, action="s1", reason="First")
        logger.log(action_type=ActionType.CLEANING, action="s2", reason="Second")
        assert logger.entry_count == 2

    def test_json_serialization(self):
        logger = AuditLogger()
        logger.log(
            action_type=ActionType.PROFILING,
            action="test",
            reason="test reason",
            details={"key": "value"},
        )
        j = logger.to_json()
        assert '"action": "test"' in j and '"reason": "test reason"' in j

    def test_clear(self):
        logger = AuditLogger()
        logger.log(action_type=ActionType.PROFILING, action="t", reason="r")
        logger.clear()
        assert logger.entry_count == 0
