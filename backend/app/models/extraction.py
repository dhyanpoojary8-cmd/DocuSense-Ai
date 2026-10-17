"""Domain models for OCR ingestion, extraction provenance, and field extraction results."""
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from app.models.field import ExtractedField


class ExtractionSource(str, Enum):
    """Source provenance of the extracted text and fields.

    Strict provenance tracking guarantees that mock fixtures or fallbacks can NEVER
    be mistaken for real OCR or digital PDF extractions.
    """
    OCR = "ocr"            # Extracted via Tesseract optical character recognition
    PDF_TEXT = "pdf_text"  # Extracted from native digital vector PDF stream
    MOCK = "mock"          # Synthetic test fixture or test mock
    VLM = "vlm"            # Optional vision-language model adapter


class ExtractedDocumentData(BaseModel):
    """Aggregated extraction result with strict provenance tracking."""
    source: ExtractionSource = Field(description="Primary extraction source: ocr, pdf_text, mock, or vlm")
    raw_text: str = Field(description="Full text stream extracted from the document")
    page_count: int = Field(default=1, ge=1, description="Number of pages ingested")
    fields: Dict[str, ExtractedField] = Field(description="Dictionary of extracted fields, each tagged with provenance")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata such as engine versions, DPI, timings")
