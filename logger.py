"""
AuditLogger - Append-only transformation audit trail.

Every data change must be traceable:
  - What changed?
  - Why did it change?
  - Who approved it?

The audit log can be serialized to JSON for database storage or file export.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Category of action that produced the audit entry."""
    PROFILING = "profiling"
    CLEANING = "cleaning"
    DUPLICATE_DETECTION = "duplicate_detection"
    ANOMALY_DETECTION = "anomaly_detection"
    HUMAN_REVIEW = "human_review"
    AI_RECOMMENDATION = "ai_recommendation"
    EXPORT = "export"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class AuditEntry(BaseModel):
    """A single auditable event."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    action_type: ActionType
    action: str
    column: Optional[str] = None
    records_affected: int = 0
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    automated: bool = True
    reviewer: Optional[str] = None


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class AuditLogger:
    """
    Append-only audit log for recording transformations.

    Usage::

        logger = AuditLogger()
        logger.log(
            action_type=ActionType.PROFILING,
            action="profile_dataset",
            reason="Initial dataset profiling completed",
            details={"rows": 1500, "columns": 12},
        )
        print(logger.to_json())
    """

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def log(
        self,
        action_type: ActionType,
        action: str,
        column: Optional[str] = None,
        records_affected: int = 0,
        reason: str = "",
        details: Optional[dict[str, Any]] = None,
        automated: bool = True,
        reviewer: Optional[str] = None,
    ) -> AuditEntry:
        """Record an audit event and return the created entry."""
        entry = AuditEntry(
            action_type=action_type,
            action=action,
            column=column,
            records_affected=records_affected,
            reason=reason,
            details=details or {},
            automated=automated,
            reviewer=reviewer,
        )
        self._entries.append(entry)
        return entry

    def get_entries(self) -> list[AuditEntry]:
        """Return all audit entries (shallow copy)."""
        return list(self._entries)

    @property
    def entry_count(self) -> int:
        """Number of entries recorded."""
        return len(self._entries)

    def to_dicts(self) -> list[dict[str, Any]]:
        """Serialize all entries to a list of dictionaries."""
        return [entry.model_dump(mode="json") for entry in self._entries]

    def to_json(self, indent: int = 2) -> str:
        """Serialize all entries to a JSON string."""
        return json.dumps(self.to_dicts(), indent=indent, default=str)

    def clear(self) -> None:
        """Clear all entries. Use with caution."""
        self._entries.clear()
