"""Domain models for extracted fields and confidence indicators."""
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field


class FieldStatus(str, Enum):
    """Integrity and validity status of an extracted field."""
    VALID = "VALID"
    MISSING = "MISSING"
    UNREADABLE = "UNREADABLE"
    AMBIGUOUS = "AMBIGUOUS"
    INCONSISTENT = "INCONSISTENT"


class ConfidenceIndicator(BaseModel):
    """Field-level confidence indicator.

    NOTE: Raw heuristic or model scores are not probability-calibrated unless
    explicitly marked `is_calibrated=True`.
    """
    score: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    is_calibrated: bool = Field(default=False, description="Whether calibration has been empirically measured")
    method: str = Field(default="heuristic", description="Scoring method: heuristic, regex_match, ocr_char_confidence, manual_correction")
    description: Optional[str] = Field(default=None, description="Human-readable explanation of score origin")


class ExtractedField(BaseModel):
    """Structured representation of a single extracted document field."""
    name: str = Field(description="Canonical field name, e.g. vendor, document_number, date, subtotal, tax, total")
    raw_value: Optional[str] = Field(default=None, description="Raw OCR or input text prior to parsing")
    normalized_value: Optional[Any] = Field(default=None, description="Parsed typed representation (e.g. ISO date string, Decimal string)")
    status: FieldStatus = Field(description="Deterministic status flag")
    confidence: ConfidenceIndicator = Field(description="Confidence/uncertainty assessment")
    flags: List[str] = Field(default_factory=list, description="Diagnostic warning/error codes")
    message: Optional[str] = Field(default=None, description="Human-readable explanation of any detected issues")
    source: str = Field(default="ocr", description="Extraction provenance: ocr, pdf_text, mock, or vlm")
