"""Deterministic rule-based field extractor with layout heuristics and provenance tagging."""
import re
from typing import Dict, List, Optional
from decimal import Decimal

from app.extraction.base import BaseFieldExtractor
from app.models.extraction import ExtractionSource
from app.models.field import ConfidenceIndicator, ExtractedField, FieldStatus
from app.validation.date_validator import parse_and_validate_date
from app.validation.money_validator import parse_monetary_value
from app.validation.rules_engine import validate_vendor, validate_document_number


# Non-vendor header words to ignore when scanning top lines
NON_VENDOR_HEADERS = {
    "tax invoice", "simplified tax invoice", "official receipt", "sales receipt",
    "cash receipt", "invoice", "receipt", "bill", "tax document", "original",
    "duplicate", "welcome", "customer copy", "merchant copy"
}


class DeterministicRuleExtractor(BaseFieldExtractor):
    """Extracts 6 core fields using regular expressions, text anchors, and layout heuristics."""

    def extract_fields(
        self,
        raw_text: str,
        source: ExtractionSource,
        currency_hint: Optional[str] = None,
    ) -> Dict[str, ExtractedField]:
        """Extract the 6 canonical document fields from text with explicit source provenance."""
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        src_tag = source.value

        fields: Dict[str, ExtractedField] = {}

        # 1. Vendor Extraction
        raw_vendor = self._extract_vendor(lines)
        if raw_vendor:
            vf = validate_vendor(raw_vendor)
            vf.source = src_tag
            fields["vendor"] = vf
        else:
            fields["vendor"] = ExtractedField(
                name="vendor",
                raw_value=None,
                normalized_value=None,
                status=FieldStatus.MISSING,
                confidence=ConfidenceIndicator(score=0.0, is_calibrated=False, method="rule_extractor", description="Vendor pattern not found"),
                flags=["MISSING_VENDOR"],
                message="Vendor not found in document text",
                source=src_tag,
            )

        # 2. Document Number Extraction
        raw_doc_num = self._extract_doc_number(lines, raw_text)
        if raw_doc_num:
            df = validate_document_number(raw_doc_num)
            df.source = src_tag
            fields["document_number"] = df
        else:
            fields["document_number"] = ExtractedField(
                name="document_number",
                raw_value=None,
                normalized_value=None,
                status=FieldStatus.MISSING,
                confidence=ConfidenceIndicator(score=0.0, is_calibrated=False, method="rule_extractor", description="Doc number pattern not found"),
                flags=["MISSING_DOCUMENT_NUMBER"],
                message="Document number not found in text",
                source=src_tag,
            )

        # 3. Date Extraction
        raw_date = self._extract_date(lines, raw_text)
        if raw_date:
            iso_d, d_status, d_flags, d_msg = parse_and_validate_date(raw_date)
            fields["date"] = ExtractedField(
                name="date",
                raw_value=raw_date,
                normalized_value=iso_d,
                status=d_status,
                confidence=ConfidenceIndicator(score=0.9 if d_status == FieldStatus.VALID else 0.4, is_calibrated=False, method="rule_extractor", description=d_msg),
                flags=d_flags,
                message=d_msg,
                source=src_tag,
            )
        else:
            fields["date"] = ExtractedField(
                name="date",
                raw_value=None,
                normalized_value=None,
                status=FieldStatus.MISSING,
                confidence=ConfidenceIndicator(score=0.0, is_calibrated=False, method="rule_extractor", description="Date not found"),
                flags=["MISSING_DATE"],
                message="Date not found in document text",
                source=src_tag,
            )

        # 4, 5, 6. Subtotal, Tax, Total Extraction
        raw_subtotal = self._extract_subtotal(lines, raw_text)
        raw_tax = self._extract_tax(lines, raw_text)
        raw_total = self._extract_total(lines, raw_text)

        for m_name, raw_val in [("subtotal", raw_subtotal), ("tax", raw_tax), ("total", raw_total)]:
            if raw_val is not None:
                dec_val, m_status, m_flags, m_msg = parse_monetary_value(
                    raw_val,
                    allow_negative=(m_name == "tax"),
                    field_name=m_name,
                    currency_hint=currency_hint,
                )
                fields[m_name] = ExtractedField(
                    name=m_name,
                    raw_value=raw_val,
                    normalized_value=str(dec_val) if dec_val is not None else None,
                    status=m_status,
                    confidence=ConfidenceIndicator(score=0.9 if m_status == FieldStatus.VALID else 0.2, is_calibrated=False, method="rule_extractor", description=m_msg),
                    flags=m_flags,
                    message=m_msg,
                    source=src_tag,
                )
            else:
                fields[m_name] = ExtractedField(
                    name=m_name,
                    raw_value=None,
                    normalized_value=None,
                    status=FieldStatus.MISSING,
                    confidence=ConfidenceIndicator(score=0.0, is_calibrated=False, method="rule_extractor", description=f"{m_name.title()} pattern not found"),
                    flags=[f"MISSING_{m_name.upper()}"],
                    message=f"{m_name.title()} not found in document text",
                    source=src_tag,
                )

        return fields

    def _extract_vendor(self, lines: List[str]) -> Optional[str]:
        """Detect vendor from labeled headers or clean top receipt lines."""
        for line in lines[:15]:
            m = re.search(r"^(?:vendor|store|merchant|company|seller)\s*[:\-]\s*(.+)$", line, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                if len(cand) >= 2:
                    return cand

        for line in lines[:5]:
            lower = line.lower()
            if any(h in lower for h in NON_VENDOR_HEADERS):
                continue
            if re.search(r"^\d+[\s\-/.]", line) or re.search(r"\b(?:tel|fax|phone|jalan|st\.|road|avenue|suite|box)\b", lower):
                continue
            if re.search(r"[a-zA-Z]{2,}", line):
                return line

        return None

    def _extract_doc_number(self, lines: List[str], full_text: str) -> Optional[str]:
        """Detect invoice number, receipt ID, or document reference."""
        # Require delimiter following keyword to avoid matching "TAX INVOICE" as doc number
        patterns = [
            r"\b(?:invoice|inv|receipt|bill|doc|order|trans)\s*(?:no\.?|num\.?|#)?[\s.:#№-]+([A-Z0-9][A-Z0-9\-_/]{2,29})\b",
            r"\b(?:check|chq)[\s.:#№-]+([0-9]{3,15})\b",
            r"\b(?:no\.?|#)[\s.:#№-]+([A-Z0-9\-_/]{4,25})\b",
        ]
        for line in lines:
            lower = line.lower()
            if any(h == lower for h in NON_VENDOR_HEADERS):
                continue
            for pat in patterns:
                m = re.search(pat, line, re.IGNORECASE)
                if m:
                    cand = m.group(1).strip()
                    if cand.lower() not in {"date", "time", "tax", "total", "subtotal"}:
                        return cand
        return None

    def _extract_date(self, lines: List[str], full_text: str) -> Optional[str]:
        """Detect transaction or invoice date."""
        for line in lines:
            if re.search(r"\bdate\b", line, re.IGNORECASE):
                m = re.search(
                    r"(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}[-/\s]+[a-zA-Z]{3,9}[-/\s,]+\d{2,4})",
                    line,
                )
                if m:
                    return m.group(1).strip()

        general_match = re.search(
            r"\b(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}[-/\s]+[a-zA-Z]{3,9}[-/\s,]+\d{2,4})\b",
            full_text,
        )
        if general_match:
            return general_match.group(1).strip()

        return None

    def _extract_subtotal(self, lines: List[str], full_text: str) -> Optional[str]:
        """Detect subtotal amount."""
        pattern = re.compile(
            r"\b(?:subtotal|sub-total|sub\s*total|jumlah|net\s*amount)(?:\s*\([^)]*\))?[\s:=-]*([$€£¥₹Rp\d\s,.]+)",
            re.IGNORECASE,
        )
        for line in lines:
            m = pattern.search(line)
            if m:
                cand = m.group(1).strip()
                cand = re.split(r"[a-zA-Z]", cand)[0].strip()
                if cand:
                    return cand
        return None

    def _extract_tax(self, lines: List[str], full_text: str) -> Optional[str]:
        """Detect tax amount (VAT, GST, Sales Tax, PB1, Pajak)."""
        pattern = re.compile(
            r"\b(?:tax|sales\s*tax|vat|gst|pajak|pb1|sst)(?:\s*\([^)]*\))?[\s:=-]*([$€£¥₹Rp\d\s,.]+)",
            re.IGNORECASE,
        )
        for line in lines:
            if re.search(r"\b(?:tax\s*invoice|tax\s*id|tax\s*no)\b", line, re.IGNORECASE):
                continue
            m = pattern.search(line)
            if m:
                cand = m.group(1).strip()
                cand = re.split(r"[a-zA-Z]", cand)[0].strip()
                if cand:
                    return cand
        return None

    def _extract_total(self, lines: List[str], full_text: str) -> Optional[str]:
        """Detect final total amount."""
        pattern = re.compile(
            r"\b(?:total\s*amount|grand\s*total|total\s*due|amount\s*due|total\s*bayar|total\s*tagihan|total)(?:\s*\([^)]*\))?[\s:=-]*([$€£¥₹Rp\d\s,.]+)",
            re.IGNORECASE,
        )
        for line in reversed(lines):
            if re.search(r"\b(?:subtotal|sub\s*total|total\s*item|total\s*qty|total\s*tax)\b", line, re.IGNORECASE):
                continue
            m = pattern.search(line)
            if m:
                cand = m.group(1).strip()
                cand = re.split(r"[a-zA-Z]", cand)[0].strip()
                if cand:
                    return cand
        return None
