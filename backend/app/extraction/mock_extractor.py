"""Explicit mock extractor for synthetic fixtures, strictly marked with source='mock'."""
from typing import Any, Dict, Optional

from app.extraction.base import BaseFieldExtractor
from app.models.extraction import ExtractionSource
from app.models.field import ConfidenceIndicator, ExtractedField, FieldStatus
from app.validation.rules_engine import validate_document_fields


class MockDocumentExtractor(BaseFieldExtractor):
    """Synthetic test fixture extractor.

    STRICT GUARANTEE:
    Every field produced by this class has source='mock', preventing synthetic or mock data
    from ever being mistaken for genuine OCR, digital PDF, or VLM extraction.
    """

    def extract_fields(
        self,
        raw_text: str,
        source: ExtractionSource = ExtractionSource.MOCK,
        currency_hint: Optional[str] = None,
    ) -> Dict[str, ExtractedField]:
        """Extract fields and guarantee source='mock'."""
        src_tag = ExtractionSource.MOCK.value
        # Parse simulated key-value text or json-like lines
        extracted: Dict[str, ExtractedField] = {}
        for line in raw_text.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                k_clean = k.strip().lower()
                v_clean = v.strip()
                if k_clean in {"vendor", "document_number", "date", "subtotal", "tax", "total"}:
                    extracted[k_clean] = ExtractedField(
                        name=k_clean,
                        raw_value=v_clean,
                        normalized_value=v_clean,
                        status=FieldStatus.VALID,
                        confidence=ConfidenceIndicator(score=1.0, is_calibrated=False, method="mock_fixture", description="Synthetic test fixture"),
                        flags=[],
                        message="Mock field",
                        source=src_tag,
                    )

        # Fill any missing mandatory fields with explicit MISSING status and mock provenance
        for req in ["vendor", "document_number", "date", "subtotal", "tax", "total"]:
            if req not in extracted:
                extracted[req] = ExtractedField(
                    name=req,
                    raw_value=None,
                    normalized_value=None,
                    status=FieldStatus.MISSING,
                    confidence=ConfidenceIndicator(score=0.0, is_calibrated=False, method="mock_fixture", description="Mock missing field"),
                    flags=[f"MISSING_{req.upper()}"],
                    message=f"Field {req} missing in mock",
                    source=src_tag,
                )

        return extracted
