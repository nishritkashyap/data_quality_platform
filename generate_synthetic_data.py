"""
Synthetic messy dataset generator for testing.

Produces realistic but fake business data with intentional quality
issues that the profiler and future cleaning engine must handle:

  - Leading / trailing whitespace
  - Mixed capitalisation (UPPER, lower, Title, mIxEd)
  - Inconsistent date formats (ISO, US, EU, short-year, invalid)
  - Valid and invalid emails
  - Inconsistent phone formats
  - Missing values (~8 % per selected column)
  - Exact duplicate rows (~5 %)
  - Inconsistent categorical values ("NY" vs "New York" vs "new york")
  - Numeric outliers and suspicious negatives
  - One completely empty column
  - One constant-value column
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "John", "Jane", "Alice", "Bob", "Charlie", "Diana",
    "Edward", "Fiona", "George", "Hannah", "Isaac", "Julia",
    "Kevin", "Laura", "Michael", "Nancy", "Oscar", "Patricia",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
    "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez",
]

MESSY_STATES = [
    "NY", "N.Y.", "New York", "new york", "NEW YORK",
    "CA", "Ca", "California", "california", "CALIFORNIA",
    "TX", "Tx", "Texas", "texas", "TEXAS",
    "FL", "florida", "Florida", "FLORIDA",
    "IL", "Illinois", "illinois",
]

MESSY_CATEGORIES = [
    "Wholesale", "Retail", "wholesale", "RETAIL",
    "retail", "WHOLESALE", "Online", "online",
]

BOOLEAN_VARIANTS = ["yes", "no", "true", "false", "Y", "N", "1", "0"]

# ---------------------------------------------------------------------------
# Generators (each returns a single cell value)
# ---------------------------------------------------------------------------

def _random_name(rng: random.Random, first: str, last: str) -> str:
    """Return a customer name with intentional whitespace / case issues."""
    style = rng.randint(0, 5)
    if style == 0:
        return f"  {first} {last}"
    if style == 1:
        return f"{first} {last}   "
    if style == 2:
        return f"{first.upper()} {last.upper()}"
    if style == 3:
        return f"{first.lower()} {last.lower()}"
    return f"{first} {last}"


def _random_email(rng: random.Random, first: str) -> str:
    """Return a valid or intentionally invalid email."""
    style = rng.randint(0, 4)
    if style == 0:
        return f"{first.lower()}.{rng.choice(LAST_NAMES).lower()}@example.com"
    if style == 1:
        return f"{first.lower()}@company.org"
    if style == 2:
        return "not-an-email"
    if style == 3:
        return f"{first.lower()}@.com"
    return f"{first.lower()}{rng.randint(1, 99)}@test.co"


def _random_phone(rng: random.Random) -> str:
    """Return a phone number in a random format (or an invalid one)."""
    area = rng.randint(200, 999)
    mid = rng.randint(100, 999)
    last4 = rng.randint(1000, 9999)
    style = rng.randint(0, 5)
    if style == 0:
        return f"({area}) {mid}-{last4}"
    if style == 1:
        return f"{area}-{mid}-{last4}"
    if style == 2:
        return f"+1{area}{mid}{last4}"
    if style == 3:
        return f"{area}.{mid}.{last4}"
    if style == 4:
        return f"{area}{mid}{last4}"
    return "12345"  # too short / invalid


def _random_date(rng: random.Random) -> str:
    """Return a date string in a random format (or an invalid one)."""
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    year = rng.randint(2019, 2025)
    style = rng.randint(0, 4)
    if style == 0:
        return f"{month:02d}/{day:02d}/{year}"
    if style == 1:
        return f"{year}-{month:02d}-{day:02d}"
    if style == 2:
        return f"{day:02d}.{month:02d}.{year}"
    if style == 3:
        return f"{month}/{day}/{year}"
    return "invalid-date"


def _random_revenue(rng: random.Random) -> float:
    """Return a revenue figure with occasional outliers / negatives."""
    r = rng.random()
    if r < 0.05:
        return round(rng.uniform(500_000, 2_000_000), 2)  # outlier
    if r < 0.08:
        return -abs(round(rng.uniform(100, 5000), 2))      # suspicious negative
    return round(rng.uniform(1000, 50_000), 2)              # normal


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_messy_customer_data(
    num_rows: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic messy customer dataset.

    Args:
        num_rows: Number of base rows (before duplicate injection).
        seed:     Random seed for reproducibility.

    Returns:
        A messy DataFrame with ~5 % duplicate rows appended.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    rows: list[dict] = []
    for i in range(num_rows):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)

        rows.append({
            "customer_id":   i + 1,
            "customer_name": _random_name(rng, first, last),
            "email":         _random_email(rng, first),
            "phone":         _random_phone(rng),
            "signup_date":   _random_date(rng),
            "state":         rng.choice(MESSY_STATES),
            "category":      rng.choice(MESSY_CATEGORIES),
            "annual_revenue": _random_revenue(rng),
            "empty_column":  None,
            "constant_flag": "ACTIVE",
            "is_verified":   rng.choice(BOOLEAN_VARIANTS),
        })

    df = pd.DataFrame(rows)

    # Inject ~8 % missing values in selected columns
    for col in ("customer_name", "email", "phone", "state", "annual_revenue"):
        mask = np.random.random(len(df)) < 0.08
        df.loc[mask, col] = None

    # Inject ~5 % exact duplicate rows
    n_dupes = max(3, num_rows // 20)
    dupe_idx = rng.sample(range(len(df)), min(n_dupes, len(df)))
    df = pd.concat([df, df.iloc[dupe_idx].copy()], ignore_index=True)

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def save_test_csv(
    filepath: str = "test_data/messy_customers.csv",
    num_rows: int = 200,
) -> Path:
    """Generate and save a messy CSV for manual inspection."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_messy_customer_data(num_rows=num_rows)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")
    return path


if __name__ == "__main__":
    save_test_csv()
