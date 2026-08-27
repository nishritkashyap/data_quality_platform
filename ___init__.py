"""
Cleaning package - Deterministic, auditable data transformations.

Every cleaner is a standalone function. The pipeline chains them together.

Usage::

    from data_engine.cleaning import CleaningPipeline

    pipeline = CleaningPipeline()
    result = pipeline.clean(df)
    print(result.cleaned_df.head())
    print(result.audit_logger.to_json())
"""

from data_engine.cleaning.pipeline import CleaningPipeline

__all__ = ["CleaningPipeline"]
