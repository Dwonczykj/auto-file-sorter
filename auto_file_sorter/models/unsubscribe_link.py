from pydantic import BaseModel
from typing import Optional


class UnsubscribeLinkOutput(BaseModel):
    """Model for AI unsubscribe link detection output"""
    link: Optional[str] = None
    location: Optional[str] = None  # 'header' or CSS selector path
    confidence: float
    reason: str  # Explanation of why the link was chosen or why none was found
