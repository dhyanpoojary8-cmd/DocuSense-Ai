"""Deterministic rule-based field extractor with layout heuristics and provenance tagging."""
import re
from typing import Dict, List, Optional
from decimal import Decimal

from app.extraction.base import BaseFieldExtractor
from app.models.extraction import ExtractionSource
from app.models.field import ConfidenceIndicator, ExtractedField, FieldStatus
from app.validation.arithmetic import verify_arithmetic
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

        parsed_decimals: Dict[str, Optional[Decimal]] = {}
        for m_name, raw_val in [("subtotal", raw_subtotal), ("tax", raw_tax), ("total", raw_total)]:
            if raw_val is not None:
                dec_val, m_status, m_flags, m_msg = parse_monetary_value(
                    raw_val,
                    allow_negative=(m_name == "tax"),
                    field_name=m_name,
                    currency_hint=currency_hint,
                )
                parsed_decimals[m_name] = dec_val
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
                parsed_decimals[m_name] = None
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

        # Cross-check rule (e): when subtotal, tax and total are all present,
        # accept them only if subtotal + tax == total within currency tolerance;
        # otherwise abstain on the inconsistent fields.
        s_d = parsed_decimals.get("subtotal")
        t_d = parsed_decimals.get("tax")
        tot_d = parsed_decimals.get("total")
        if s_d is not None and t_d is not None and tot_d is not None:
            arith_check = verify_arithmetic(s_d, t_d, tot_d, currency=currency_hint)
            if not arith_check.is_valid:
                for fn in ("subtotal", "tax", "total"):
                    fields[fn].status = FieldStatus.INCONSISTENT
                    fields[fn].normalized_value = None
                    if "ARITHMETIC_MISMATCH" not in fields[fn].flags:
                        fields[fn].flags.append("ARITHMETIC_MISMATCH")
                    fields[fn].message = f"Arithmetic mismatch ({s_d} + {t_d} != {tot_d}); abstaining on inconsistent field"
                    fields[fn].confidence.score = 0.1
                    fields[fn].confidence.description = fields[fn].message

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

    def _extract_money_field(
        self,
        lines: List[str],
        keywords: List[str],
        exclude_line_regex: Optional[re.Pattern] = None,
        reverse: bool = False,
        allow_next_line: bool = True,
        truncate_at_cash: bool = False,
    ) -> Optional[str]:
        """Find money field matching keywords with OCR tolerance and next-line fallback."""
        indexed_lines = list(reversed(list(enumerate(lines)))) if reverse else list(enumerate(lines))

        # Pass 1: Look for exact keyword match first
        for i, line in indexed_lines:
            if exclude_line_regex and exclude_line_regex.search(line):
                continue

            target_text = line
            if truncate_at_cash:
                m_cash = re.search(r"\b(?:cash|tunai|change|kembali|kembalian)\b", line, re.IGNORECASE)
                if m_cash:
                    target_text = line[: m_cash.start()].strip()
                    if not target_text:
                        continue

            if _exact_match_keyword(target_text, keywords):
                cands = _extract_amount_candidates(target_text)
                if cands:
                    return cands[-1] if reverse else cands[0]

                if allow_next_line and i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if not re.search(r"\b(?:total|subtotal|sub\s*total|tax|cash|tunai|kembali|change|invoice|inv|date)\b", next_line, re.IGNORECASE):
                        next_cands = _extract_amount_candidates(next_line)
                        if next_cands:
                            return next_cands[0]

        # Pass 2: Fuzzy keyword match fallback
        for i, line in indexed_lines:
            if exclude_line_regex and exclude_line_regex.search(line):
                continue

            target_text = line
            if truncate_at_cash:
                m_cash = re.search(r"\b(?:cash|tunai|change|kembali|kembalian)\b", line, re.IGNORECASE)
                if m_cash:
                    target_text = line[: m_cash.start()].strip()
                    if not target_text:
                        continue

            if _fuzzy_match_keyword(target_text, keywords):
                cands = _extract_amount_candidates(target_text)
                if cands:
                    return cands[-1] if reverse else cands[0]

                if allow_next_line and i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if not re.search(r"\b(?:total|subtotal|sub\s*total|tax|cash|tunai|kembali|change|invoice|inv|date)\b", next_line, re.IGNORECASE):
                        next_cands = _extract_amount_candidates(next_line)
                        if next_cands:
                            return next_cands[0]

        return None

    def _extract_subtotal(self, lines: List[str], full_text: str) -> Optional[str]:
        """Detect subtotal amount with OCR tolerance and next-line fallback."""
        keywords = ["subtotal", "sub total", "sub-total", "jumlah", "net amount", "sub l total"]
        exclude = re.compile(r"\b(?:tax\s*invoice)\b", re.IGNORECASE)
        return self._extract_money_field(lines, keywords, exclude_line_regex=exclude, reverse=False)

    def _extract_tax(self, lines: List[str], full_text: str) -> Optional[str]:
        """Detect tax amount (VAT, GST, Sales Tax, PB1, PPN, Pajak)."""
        keywords = ["tax", "pb1", "pb 1", "ppn", "pajak", "vat", "gst", "sales tax", "sst"]
        exclude = re.compile(r"\b(?:tax\s*invoice|tax\s*id|tax\s*no)\b", re.IGNORECASE)
        return self._extract_money_field(lines, keywords, exclude_line_regex=exclude, reverse=False)

    def _extract_total(self, lines: List[str], full_text: str) -> Optional[str]:
        """Detect final total amount excluding cash/tunai/change lines."""
        keywords = ["total", "grand total", "total amount", "total due", "amount due", "total bayar", "total tagihan"]
        exclude = re.compile(r"\b(?:sub\s*[a-z]?\s*total|total\s*item|total\s*qty|total\s*tax)\b", re.IGNORECASE)
        return self._extract_money_field(lines, keywords, exclude_line_regex=exclude, reverse=True, truncate_at_cash=True)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Standard Levenshtein distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1] * (len(s2) + 1)
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
        prev = curr
    return prev[-1]


def _exact_match_keyword(text: str, keywords: List[str]) -> bool:
    """Exact whitespace-aware regex match for keywords."""
    for kw in keywords:
        pattern = r"\b" + re.escape(kw).replace(r"\ ", r"[\s\-_]*") + r"\b"
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _fuzzy_match_keyword(text: str, keywords: List[str]) -> bool:
    """Whitespace-insensitive and OCR edit-distance tolerant keyword matching."""
    words = text.split()
    for w in words:
        clean_w = re.sub(r"[^a-zA-Z0-9]", "", w.lower())
        if not clean_w:
            continue
        for kw in keywords:
            clean_kw = re.sub(r"[^a-zA-Z0-9]", "", kw.lower())
            if not clean_kw:
                continue
            if len(clean_kw) <= 3:
                max_d = 0 if clean_kw != "pb1" else 1
            elif len(clean_kw) <= 4 or clean_kw == "pajak":
                max_d = 1
            else:
                max_d = 2

            if abs(len(clean_w) - len(clean_kw)) <= max_d:
                if levenshtein_distance(clean_w, clean_kw) <= max_d:
                    return True

    # 2-word and 3-word n-grams for phrases like "sub l total" or "pb 1"
    for n in (2, 3):
        for i in range(len(words) - n + 1):
            ngram = re.sub(r"[^a-zA-Z0-9]", "", "".join(words[i : i + n]).lower())
            for kw in keywords:
                clean_kw = re.sub(r"[^a-zA-Z0-9]", "", kw.lower())
                if not clean_kw or len(clean_kw) <= 3:
                    continue
                max_d = 1 if (len(clean_kw) <= 4 or clean_kw == "pajak") else 2
                if abs(len(ngram) - len(clean_kw)) <= max_d:
                    if levenshtein_distance(ngram, clean_kw) <= max_d:
                        return True

    return False


def _extract_amount_candidates(line: str) -> List[str]:
    """Extract candidate currency amounts from a receipt line."""
    cleaned = re.sub(r"\d+\s*%", "", line)
    cleaned = re.sub(r"\b\d+\s*[xX]\b", "", cleaned)
    cands = []
    # Match amounts with currency symbols or standard digits/separators
    pattern = r"(?:[\$€£¥₹]|Rp\.?|IDR)?\s*(?:\d{1,3}(?:\s*[,.]\s*\d{3})+(?:[.,]\d{2})?|\d+[.,]\d{2}|\b\d{4,}\b|\b0\b)"
    for m in re.finditer(pattern, cleaned, re.IGNORECASE):
        val = m.group(0).strip().rstrip(".,-:")
        if val:
            cands.append(val)
    return cands
