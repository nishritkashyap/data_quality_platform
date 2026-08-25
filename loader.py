"""
Data Loader - Secure CSV & XLSX file ingestion.

Validates file type, enforces size limits, and loads data into
a Pandas DataFrame with all values preserved as strings (raw mode).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_FILE_SIZE_MB: int = 100
ALLOWED_EXTENSIONS: set[str] = {".csv", ".xlsx", ".xls"}

NA_VALUES: list[str] = [
    "", "NA", "N/A", "null", "NULL",
    "None", "none", "nan", "NaN", "#N/A", "#NA",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DataLoadError(Exception):
    """Raised when a file cannot be safely loaded."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_file(filepath: Union[str, Path]) -> Path:
    """
    Validate file existence, extension, and size.

    Returns:
        Resolved ``Path`` on success.

    Raises:
        DataLoadError: On any validation failure.
    """
    path = Path(filepath).resolve()

    if not path.exists():
        raise DataLoadError(f"File not found: {path}")

    if not path.is_file():
        raise DataLoadError(f"Not a file: {path}")

    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise DataLoadError(
            f"Unsupported file type: '{ext}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise DataLoadError(
            f"File too large: {size_mb:.1f} MB. "
            f"Maximum allowed: {MAX_FILE_SIZE_MB} MB"
        )

    return path


def load_dataframe(
    filepath: Union[str, Path],
    sheet_name: Optional[Union[str, int]] = 0,
) -> pd.DataFrame:
    """
    Load a CSV or XLSX file into a Pandas DataFrame.

    All values are loaded as strings to preserve the raw data exactly
    as it appears in the file. Type inference happens later in profiling.

    Args:
        filepath:   Path to the CSV or XLSX file.
        sheet_name: For Excel files, which sheet to load (default: first).

    Returns:
        Raw DataFrame (unmodified).

    Raises:
        DataLoadError: If validation fails or the file cannot be parsed.
    """
    path = validate_file(filepath)
    ext = path.suffix.lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(
                path,
                dtype=str,
                keep_default_na=True,
                na_values=NA_VALUES,
            )
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(
                path,
                sheet_name=sheet_name,
                dtype=str,
                keep_default_na=True,
                na_values=NA_VALUES,
            )
        else:
            raise DataLoadError(f"Unsupported extension: {ext}")

    except DataLoadError:
        raise
    except Exception as e:
        raise DataLoadError(f"Failed to parse file: {e}") from e

    if df.empty and len(df.columns) == 0:
        raise DataLoadError("File contains no data and no column headers.")

    return df


def get_file_metadata(filepath: Union[str, Path]) -> dict:
    """Return basic file metadata without loading the data."""
    path = validate_file(filepath)
    stat = path.stat()
    return {
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
    }
