"""
DataPure - AI-Assisted Data Quality & Data Cleaning Engine

Usage:
    from data_engine import DataProfiler, CleaningPipeline, AuditLogger

    profiler = DataProfiler()
    profile = profiler.profile_file("data.csv")

    pipeline = CleaningPipeline()
    output = pipeline.clean(df)
"""

__version__ = "0.1.0"

from data_engine.audit.logger import ActionType, AuditLogger
from data_engine.cleaning.pipeline import CleaningConfig, CleaningPipeline
from data_engine.profiling.profiler import DataProfiler

__all__ = [
    "DataProfiler",
    "CleaningPipeline",
    "CleaningConfig",
    "AuditLogger",
    "ActionType",
]

