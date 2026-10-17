"""Optional Vision-Language Model (VLM) adapter, completely isolated and OFF by default."""
import os
from typing import Any, Dict, Optional
from pathlib import Path

from app.models.extraction import ExtractionSource
from app.models.field import ExtractedField


class VisionModelAdapter:
    """Optional VLM adapter for zero-shot or multimodal document parsing.

    SECURITY & INTEGRITY RULES:
    1. STRICTLY OFF BY DEFAULT: Only enabled if ENABLE_VLM=true AND an API key is present.
    2. NEVER INVENT RESULTS: If disabled, key missing, or request fails, returns status='not_run'.
    3. EXPLICIT PROVENANCE: Results are tagged with source='vlm'.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        enabled: Optional[bool] = None,
    ):
        self.api_key = api_key or os.environ.get("VLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        # Default to False unless explicitly set in environment
        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = os.environ.get("ENABLE_VLM", "false").lower() in ("true", "1", "yes")
        self.model_name = model_name

    def is_active(self) -> bool:
        """Return True only if VLM is explicitly enabled and API key is present."""
        return bool(self.enabled and self.api_key)

    def extract_document(
        self,
        document_path: Path,
        mime_type: str = "image/png",
    ) -> Dict[str, Any]:
        """Extract fields using multimodal VLM if active, or report 'not_run' without inventing results."""
        if not self.is_active():
            return {
                "status": "not_run",
                "source": ExtractionSource.VLM.value,
                "reason": "VLM adapter is disabled by default. Set ENABLE_VLM=true and configure VLM_API_KEY in .env to enable.",
                "fields": {},
            }

        # Placeholder for external HTTP call (e.g. OpenAI / Gemini vision endpoints)
        # Note: If called in offline mode or without working keys, strictly report failure rather than fabricating data
        return {
            "status": "not_run",
            "source": ExtractionSource.VLM.value,
            "reason": "VLM integration configured but remote endpoint call not initiated in offline test mode.",
            "fields": {},
        }
