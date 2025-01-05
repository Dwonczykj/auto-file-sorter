from pydantic import BaseModel
from typing import Optional


class ArchiveDecisionOutput(BaseModel):
    """Model for AI archive decision output"""
    can_archive: bool
    confidence: float
    reason: str
    importance_score: float  # 0-1 score of email importance
    summary: Optional[str] = None  # Brief summary if email seems important
