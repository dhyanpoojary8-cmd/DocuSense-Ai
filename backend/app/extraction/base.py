"""Abstract base classes for OCR engines, PDF extractors, and field extraction rules."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image

from app.models.extraction import ExtractedDocumentData, ExtractionSource
from app.models.field import ExtractedField


class BaseOCREngine(ABC):
    """Abstract interface for Optical Character Recognition engines."""

    @abstractmethod
    def extract_text_from_image(self, image: Union[str, Path, Image.Image]) -> str:
        """Extract plain text string from an image file or PIL Image object."""
        pass

    @abstractmethod
    def get_source_identifier(self) -> ExtractionSource:
        """Return the immutable extraction source identifier (e.g. ExtractionSource.OCR)."""
        pass


class BasePDFExtractor(ABC):
    """Abstract interface for PDF document processors."""

    @abstractmethod
    def extract(self, pdf_path: Union[str, Path]) -> ExtractedDocumentData:
        """Extract text and metadata from PDF, routing to digital text or OCR as appropriate."""
        pass


class BaseFieldExtractor(ABC):
    """Abstract interface for extracting key fields from document text."""

    @abstractmethod
    def extract_fields(
        self,
        raw_text: str,
        source: ExtractionSource,
        currency_hint: Optional[str] = None,
    ) -> Dict[str, ExtractedField]:
        """Extract the 6 canonical document fields from text with explicit source provenance."""
        pass
