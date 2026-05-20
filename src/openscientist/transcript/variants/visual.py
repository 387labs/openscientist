"""Visual TranscriptEntry variants: image view, image generation."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ImageView(BaseModel):
    """Codex ``ImageViewThreadItem``."""

    type: Literal["image_view"] = "image_view"
    id: str
    path: str
    raw: dict[str, Any] = Field(default_factory=dict)


class ImageGeneration(BaseModel):
    """Codex ``ImageGenerationThreadItem``."""

    type: Literal["image_generation"] = "image_generation"
    id: str
    result: str
    revised_prompt: str | None = None
    saved_path: str | None = None
    status: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)
