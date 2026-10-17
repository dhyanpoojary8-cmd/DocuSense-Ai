"""Comprehensive tests for Stage 2 OCR and rule-based extraction pipeline.

Covers:
- Tesseract OCR binary resolution and OCR execution on synthetic PIL image
- Native digital PDF text extraction (source='pdf_text')
- Scanned PDF rendering via pypdfium2 and OCR routing (source='ocr')
- Deterministic rule-based extraction of 6 fields with strict provenance
- Currency/locale aware parsing for Indonesian Rupiah (IDR 99,000 -> 99000) and zero-decimal arithmetic
- VisionModelAdapter off-by-default behavior and honest 'not_run' reporting
- MockDocumentExtractor ensuring mock data is always explicitly marked source='mock'
"""
from decimal import Decimal
from pathlib import Path
import pytest
from PIL import Image, ImageDraw

from app.extraction.tesseract_engine import TesseractOCREngine, resolve_tesseract_binary
from app.extraction.pdf_extractor import PDFDocumentExtractor
from app.extraction.rule_extractor import DeterministicRuleExtractor
from app.extraction.vision_adapter import VisionModelAdapter
from app.extraction.mock_extractor import MockDocumentExtractor
from app.models.extraction import ExtractionSource
from app.models.field import FieldStatus
from app.models.validation import ArithmeticStatus
from app.validation.arithmetic import verify_arithmetic
from app.validation.money_validator import parse_monetary_value


def _create_minimal_digital_pdf(path: Path, text: str) -> None:
    """Helper to create a valid minimal Type1 Helvetica vector PDF with text stream."""
    stream_content = f"BT /F1 14 Tf 50 700 Td ({text}) Tj ET".encode("latin1")
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
        b"2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj\n"
        b"3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources <</Font <</F1 5 0 R>>>>>> endobj\n"
        b"4 0 obj <</Length " + str(len(stream_content)).encode("ascii") + b">> stream\n"
        + stream_content + b"\nendstream endobj\n"
        b"5 0 obj <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>> endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"trailer <</Size 6 /Root 1 0 R>>\nstartxref\n0\n%%EOF"
    )
    path.write_bytes(pdf_bytes)


# ============================================================================
# 1. TESSERACT ENGINE TESTS
# ============================================================================

class TestTesseractEngine:
    """Tests for local Tesseract OCR integration."""

    def test_tesseract_binary_found(self):
        cmd = resolve_tesseract_binary()
        assert cmd is not None, "Tesseract executable path was not resolved"
        assert Path(cmd).is_file(), f"Tesseract path '{cmd}' does not exist"

    def test_tesseract_ocr_on_synthetic_image(self, tmp_path):
        """Render high-contrast text onto a PIL image and assert Tesseract extracts it."""
        img = Image.new("RGB", (600, 150), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        # Render text with slight thickness for crisp OCR recognition
        for offset in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            draw.text((30 + offset[0], 50 + offset[1]), "DOCUSENSE INVOICE TEST", fill=(0, 0, 0))

        img_path = tmp_path / "test_ocr.png"
        img.save(img_path)

        engine = TesseractOCREngine()
        text = engine.extract_text_from_image(img_path)
        assert "DOCUSENSE" in text.upper()
        assert "INVOICE" in text.upper()
        assert engine.get_source_identifier() == ExtractionSource.OCR


# ============================================================================
# 2. PDF EXTRACTOR TESTS (DIGITAL VS SCANNED OCR)
# ============================================================================

class TestPDFExtractor:
    """Tests for dual-mode PDF routing."""

    def test_digital_vector_pdf_extraction(self, tmp_path):
        """A digital PDF with embedded text stream should report source='pdf_text'."""
        pdf_path = tmp_path / "digital_sample.pdf"
        _create_minimal_digital_pdf(pdf_path, "INVOICE ABC-9988 VENDOR: ACME CORP TOTAL: $150.00")

        extractor = PDFDocumentExtractor()
        text, source, pages = extractor.extract_from_file(pdf_path)
        assert "INVOICE ABC-9988" in text
        assert source == ExtractionSource.PDF_TEXT
        assert pages == 1

    def test_scanned_image_pdf_extraction(self, tmp_path):
        """A PDF containing only a raster image should be rasterized via pypdfium2 and report source='ocr'."""
        # 1. Create a raster image with clear text
        img = Image.new("RGB", (600, 150), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        for offset in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            draw.text((30 + offset[0], 50 + offset[1]), "SCANNED RECEIPT TOTAL $42.00", fill=(0, 0, 0))

        # 2. Save it as a PDF (image encapsulated in PDF without digital font stream)
        pdf_path = tmp_path / "scanned_sample.pdf"
        img.save(pdf_path, "PDF")

        extractor = PDFDocumentExtractor()
        text, source, pages = extractor.extract_from_file(pdf_path)
        assert source == ExtractionSource.OCR
        assert pages == 1
        assert "SCANNED" in text.upper() or "RECEIPT" in text.upper()


# ============================================================================
# 3. RULE-BASED FIELD EXTRACTOR & PROVENANCE
# ============================================================================

class TestDeterministicRuleExtractor:
    """Tests for regex and layout heuristic extraction of the 6 core fields."""

    def test_extract_all_six_fields(self):
        sample_invoice_text = """
        TAX INVOICE
        Global Logistics Services Sdn Bhd
        No. 12, Industrial Park, Selangor
        Invoice No: INV-2024-88412
        Date: 15/10/2024

        Description        Qty    Amount
        Consulting Hours    10    100.00
        Software License     1    400.00

        Subtotal: $500.00
        Sales Tax: $40.00
        Grand Total: $540.00
        """
        extractor = DeterministicRuleExtractor()
        fields = extractor.extract_fields(sample_invoice_text, source=ExtractionSource.OCR)

        assert fields["vendor"].normalized_value == "Global Logistics Services Sdn Bhd"
        assert fields["vendor"].source == "ocr"

        assert fields["document_number"].raw_value == "INV-2024-88412"
        assert fields["document_number"].normalized_value == "2024-88412"
        assert fields["document_number"].source == "ocr"

        assert fields["date"].normalized_value == "2024-10-15"
        assert fields["date"].status == FieldStatus.VALID
        assert fields["date"].source == "ocr"

        assert fields["subtotal"].normalized_value == "500.00"
        assert fields["tax"].normalized_value == "40.00"
        assert fields["total"].normalized_value == "540.00"
        assert fields["total"].source == "ocr"

    def test_provenance_marking_pdf_text(self):
        """When text comes from digital PDF, all extracted fields must carry source='pdf_text'."""
        text = "Acme Tools\nInvoice No: 12345\nDate: 2024-01-01\nTotal: 10.00"
        extractor = DeterministicRuleExtractor()
        fields = extractor.extract_fields(text, source=ExtractionSource.PDF_TEXT)

        for name, field in fields.items():
            assert field.source == "pdf_text", f"Field {name} had source '{field.source}' instead of 'pdf_text'"


# ============================================================================
# 4. CURRENCY & LOCALE AWARENESS (IDR ZERO-DECIMAL PARSING)
# ============================================================================

class TestCurrencyAwarenessAndIDR:
    """Tests for requirement (5): IDR 99,000 parsed as integer 99000 with currency-aware tolerance."""

    def test_idr_thousand_comma_parsed_as_integer(self):
        val, status, flags, msg = parse_monetary_value("99,000", currency_hint="IDR")
        assert status == FieldStatus.VALID
        assert val == Decimal("99000")
        assert "99000" in msg

    def test_idr_thousand_dot_parsed_as_integer(self):
        val, status, flags, msg = parse_monetary_value("Rp 108.900", currency_hint="IDR")
        assert status == FieldStatus.VALID
        assert val == Decimal("108900")

    def test_idr_cord_receipt_extraction_and_arithmetic(self):
        """CORD-style receipt text with Subtotal 99,000, Tax 9,900, Total 108,900."""
        cord_text = """
        Warung Makan Sederhana
        Receipt No: 44102
        Date: 12/08/2023
        
        Nasi Campur Bali    75,000
        Ice Lemon Tea       24,000
        
        Subtotal: 99,000
        PB1 (Tax): 9,900
        Total: 108,900
        """
        extractor = DeterministicRuleExtractor()
        fields = extractor.extract_fields(cord_text, source=ExtractionSource.OCR, currency_hint="IDR")

        assert fields["subtotal"].normalized_value == "99000"
        assert fields["tax"].normalized_value == "9900"
        assert fields["total"].normalized_value == "108900"

        # Arithmetic verification in IDR
        sub_dec = Decimal(fields["subtotal"].normalized_value)
        tax_dec = Decimal(fields["tax"].normalized_value)
        tot_dec = Decimal(fields["total"].normalized_value)

        res = verify_arithmetic(sub_dec, tax_dec, tot_dec, currency="IDR")
        assert res.is_valid is True
        assert res.status == ArithmeticStatus.BALANCED
        assert res.difference == "0"

    def test_idr_tolerance_allows_small_hundred_rupiah_rounding(self):
        """In IDR, a difference of 50 Rupiah is accepted within default 100 IDR tolerance."""
        sub = Decimal("99000")
        tax = Decimal("9950")
        tot = Decimal("108900")  # Difference is 50 Rupiah
        res = verify_arithmetic(sub, tax, tot, currency="IDR")
        assert res.is_valid is True
        assert res.status == ArithmeticStatus.TOLERANCE_ACCEPTED


# ============================================================================
# 5. VISION MODEL ADAPTER (OFF BY DEFAULT)
# ============================================================================

class TestVisionModelAdapter:
    """Tests ensuring VLM adapter is off by default and reports 'not_run' honestly."""

    def test_vlm_off_by_default(self):
        vlm = VisionModelAdapter()
        assert vlm.is_active() is False

    def test_vlm_reports_not_run_when_disabled(self, tmp_path):
        dummy_file = tmp_path / "receipt.png"
        dummy_file.write_bytes(b"fake image bytes")

        vlm = VisionModelAdapter(enabled=False)
        result = vlm.extract_document(dummy_file)
        assert result["status"] == "not_run"
        assert result["source"] == "vlm"
        assert "disabled by default" in result["reason"]
        assert result["fields"] == {}


# ============================================================================
# 6. MOCK EXTRACTOR PROVENANCE INTEGRITY
# ============================================================================

class TestMockExtractorProvenance:
    """Tests ensuring mock data is strictly labeled source='mock' and can never be confused with real OCR."""

    def test_mock_extractor_always_stamps_mock_source(self):
        mock_text = """
        vendor: Fake Store
        document_number: MOCK-001
        date: 2024-01-01
        subtotal: 50.00
        tax: 5.00
        total: 55.00
        """
        extractor = MockDocumentExtractor()
        fields = extractor.extract_fields(mock_text)

        assert len(fields) == 6
        for name, field in fields.items():
            assert field.source == "mock", f"Field {name} was not marked with source='mock'!"
