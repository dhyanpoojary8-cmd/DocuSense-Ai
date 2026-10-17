"""Dual-mode PDF and image text extractor routing between digital text and Tesseract OCR."""
from pathlib import Path
from typing import Optional, Tuple, Union
import pypdf
import pypdfium2

from app.extraction.base import BasePDFExtractor
from app.extraction.tesseract_engine import TesseractOCREngine
from app.models.extraction import ExtractedDocumentData, ExtractionSource


class PDFDocumentExtractor(BasePDFExtractor):
    """Processes PDF and image files, accurately recording whether extraction was digital or OCR."""

    def __init__(self, ocr_engine: Optional[TesseractOCREngine] = None):
        self.ocr_engine = ocr_engine or TesseractOCREngine()

    def extract_from_file(self, file_path: Union[str, Path]) -> Tuple[str, ExtractionSource, int]:
        """Extract text from PDF or image file with exact provenance tagging.

        Returns:
            (extracted_text, source, page_count)
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}:
            ocr_text = self.ocr_engine.extract_text_from_image(path)
            return ocr_text, ExtractionSource.OCR, 1

        if ext == ".pdf":
            return self._extract_from_pdf(path)

        raise ValueError(f"Unsupported document format: {ext}")

    def _extract_from_pdf(self, pdf_path: Path) -> Tuple[str, ExtractionSource, int]:
        """Extract text from PDF, preferring native digital text and falling back to OCR rasterization."""
        # Step 1: Attempt digital vector text extraction with pypdf
        try:
            reader = pypdf.PdfReader(str(pdf_path))
            page_count = len(reader.pages)
            digital_pages = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                digital_pages.append(page_text.strip())

            full_digital_text = "\n\n".join(digital_pages).strip()
            # If the PDF has substantial extractable text (>= 20 alphanumeric chars), use digital text
            alphanumeric_count = sum(c.isalnum() for c in full_digital_text)
            if alphanumeric_count >= 20:
                return full_digital_text, ExtractionSource.PDF_TEXT, page_count
        except Exception:
            # If digital extraction fails or raises, fall back to OCR rasterization
            page_count = 1

        # Step 2: Fall back to rasterization via pypdfium2 and OCR with Tesseract
        pdf = pypdfium2.PdfDocument(str(pdf_path))
        page_count = len(pdf)
        ocr_pages = []

        for page_idx in range(page_count):
            page = pdf[page_idx]
            # Render page at 300 DPI for high OCR fidelity (scale = 300 / 72 ≈ 4.167)
            bitmap = page.render(scale=300 / 72)
            pil_image = bitmap.to_pil()
            page_text = self.ocr_engine.extract_text_from_image(pil_image)
            if page_text.strip():
                ocr_pages.append(page_text.strip())

        full_ocr_text = "\n\n".join(ocr_pages).strip()
        return full_ocr_text, ExtractionSource.OCR, page_count

    def extract(self, pdf_path: Union[str, Path]) -> ExtractedDocumentData:
        """Extract document text and return ExtractedDocumentData schema."""
        text, source, pages = self.extract_from_file(pdf_path)
        return ExtractedDocumentData(
            source=source,
            raw_text=text,
            page_count=pages,
            fields={},
            metadata={"file_path": str(pdf_path)},
        )
