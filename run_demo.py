"""
Demo Runner - Generate messy data, profile it, and print a summary.

Usage:
    python run_demo.py
"""

from __future__ import annotations

import json

from data_engine import ActionType, AuditLogger, DataProfiler
from data_engine.schemas import ColumnDataType
from tests.generate_synthetic_data import generate_messy_customer_data


def _separator(title: str, char: str = "=", width: int = 70) -> None:
    print(f"\n{char * width}")
    print(f"  {title}")
    print(char * width)


def main() -> None:
    _separator("DataPure - Data Profiling Engine Demo")

    # 1. Generate messy data ───────────────────────────────────────────────
    print("\n[1] Generating synthetic messy customer dataset...")
    df = generate_messy_customer_data(num_rows=200, seed=42)
    print(f"    Generated {len(df)} rows x {len(df.columns)} columns")

    # 2. Profile (read-only) ───────────────────────────────────────────────
    print("\n[2] Profiling dataset (original data is NOT modified)...")
    profiler = DataProfiler()
    audit = AuditLogger()
    profile = profiler.profile_dataframe(df, filename="messy_customers.csv")

    audit.log(
        action_type=ActionType.PROFILING,
        action="profile_dataset",
        reason="Initial dataset profiling completed",
        details={
            "rows": profile.total_rows,
            "columns": profile.total_columns,
            "duplicates": profile.exact_duplicate_rows,
            "missing_cells": profile.total_missing_cells,
        },
    )

    # 3. Summary ───────────────────────────────────────────────────────────
    _separator("PROFILING RESULTS")

    print(f"\n  Dataset:          {profile.filename}")
    print(f"  Rows:             {profile.total_rows}")
    print(f"  Columns:          {profile.total_columns}")
    print(f"  Completeness:     {profile.overall_completeness_percentage}%")
    print(f"  Missing Cells:    {profile.total_missing_cells} / {profile.total_cells}")
    print(f"  Exact Duplicates: {profile.exact_duplicate_rows} ({profile.exact_duplicate_percentage}%)")

    if profile.empty_columns:
        print(f"  Empty Columns:    {', '.join(profile.empty_columns)}")
    if profile.constant_columns:
        print(f"  Constant Columns: {', '.join(profile.constant_columns)}")

    print(f"\n  Column Types:")
    print(f"    Numeric:  {profile.numeric_column_count}")
    print(f"    Text:     {profile.text_column_count}")
    print(f"    Date:     {profile.date_column_count}")
    print(f"    Email:    {profile.email_column_count}")
    print(f"    Phone:    {profile.phone_column_count}")
    print(f"    Boolean:  {profile.boolean_column_count}")

    # 4. Per-column details ────────────────────────────────────────────────
    _separator("COLUMN DETAILS", char="-")

    for col in profile.columns:
        issues: list[str] = []
        if col.missing_count > 0:
            issues.append(f"{col.missing_count} missing")
        if col.leading_trailing_whitespace_count > 0:
            issues.append(f"{col.leading_trailing_whitespace_count} whitespace")
        if col.mixed_case_count > 0:
            issues.append(f"{col.mixed_case_count} mixed case")
        if col.detected_type == ColumnDataType.EMPTY:
            issues.append("EMPTY COLUMN")
        if col.detected_type == ColumnDataType.CONSTANT:
            issues.append("CONSTANT VALUE")

        print(f"\n  [{col.column_index}] {col.column_name}")
        print(f"      Type:     {col.detected_type.value}")
        print(f"      Unique:   {col.unique_count} ({col.unique_percentage}%)")
        print(f"      Issues:   {' | '.join(issues) if issues else 'No issues'}")

        if col.numeric_stats:
            s = col.numeric_stats
            print(f"      Stats:    min={s.min}, max={s.max}, mean={s.mean}, std={s.std}")
            print(f"                Q1={s.q1}, Q3={s.q3}, IQR={s.iqr}")
            if s.negative_count > 0:
                print(f"                [!] {s.negative_count} negative values")

        for fp in col.format_patterns:
            print(f"      Pattern:  {fp.pattern_name} ({fp.match_percentage}% match)")

        if col.is_potential_id:
            print(f"      [*] Potential identifier column")

    # 5. Audit log ─────────────────────────────────────────────────────────
    _separator("AUDIT LOG", char="-")

    for entry in audit.get_entries():
        print(f"\n  [{entry.timestamp.strftime('%H:%M:%S')}] {entry.action}")
        print(f"    Reason:  {entry.reason}")
        print(f"    Details: {json.dumps(entry.details, indent=6)}")

    _separator("Phase 1 profiling complete. Original data was NOT modified.")


if __name__ == "__main__":
    main()
