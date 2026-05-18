"""Codex review-mode TranscriptEntry variants: ReviewModeEntered, ReviewModeExited."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ReviewModeEntered(BaseModel):
    """Codex ``EnteredReviewModeThreadItem``."""

    type: Literal["review_mode_entered"] = "review_mode_entered"
    id: str
    review: str
    raw: dict[str, Any] = Field(default_factory=dict)


class ReviewModeExited(BaseModel):
    """Codex ``ExitedReviewModeThreadItem``."""

    type: Literal["review_mode_exited"] = "review_mode_exited"
    id: str
    review: str
    raw: dict[str, Any] = Field(default_factory=dict)
