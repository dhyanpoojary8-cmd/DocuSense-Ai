"""Master deterministic validation engine orchestrating field and arithmetic verification."""
import re
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Optional

from app.models.field import ConfidenceIndicator, ExtractedField, FieldStatus
from app.models.validation import ArithmeticStatus, DocumentValidationResult
from app.validation.arithmetic import verify_arithmetic, DEFAULT_TOLERANCE
from app.validation.date_validator import parse_and_validate_date
from app.validation.money_validator import parse_monetary_value


MANDATORY_FIELDS = ["vendor", "document_number", "date", "subtotal", "tax", "total"]


def validate_vendor(raw_value: Optional[str]) -> ExtractedField:
    """Validate vendor name integrity without inventing names."""
    if raw_value is None:
        return ExtractedField(
            name="vendor",
            raw_value=None,
            normalized_value=None,
            status=FieldStatus.MISSING,
            confidence=ConfidenceIndicator(score=0.0, is_calibrated=False, method="heuristic", description="Vendor missing"),
            flags=["MISSING_VENDOR"],
            message="Vendor name is missing from document",
        )

    cleaned = str(raw_value).strip()
    if not cleaned:
        return ExtractedField(
            name="vendor",
            raw_value=raw_value,
            normalized_value=None,
            status=FieldStatus.MISSING,
            confidence=ConfidenceIndicator(score=0.0, is_calibrated=False, method="heuristic", description="Vendor empty"),
            flags=["MISSING_VENDOR"],
            message="Vendor name is empty",
        )

    # Check for unreadable gibberish (e.g. "###$$$@@", strings with no letters/digits, or OCR error artifacts)
    if not re.search(r"[a-zA-Z0-9]", cleaned) or len(cleaned) < 2:
        return ExtractedField(
            name="vendor",
            raw_value=raw_value,
            normalized_value=None,
            status=FieldStatus.UNREADABLE,
            confidence=ConfidenceIndicator(score=0.2, is_calibrated=False, method="heuristic", description="OCR gibberish"),
            flags=["UNREADABLE_VENDOR"],
            message=f"Vendor text '{cleaned}' appears unreadable or noisy",
        )

    # Clean multi-spaces
    normalized = re.sub(r"\s+", " ", cleaned).title()
    return ExtractedField(
        name="vendor",
        raw_value=raw_value,
        normalized_value=normalized,
        status=FieldStatus.VALID,
        confidence=ConfidenceIndicator(score=0.9, is_calibrated=False, method="heuristic", description="Valid text string"),
        flags=[],
        message="Vendor recognized",
    )


def validate_document_number(raw_value: Optional[str]) -> ExtractedField:
    """Validate invoice/receipt/document number without inventing identifiers."""
    if raw_value is None:
        return ExtractedField(
            name="document_number",
            raw_value=None,
            normalized_value=None,
            status=FieldStatus.MISSING,
            confidence=ConfidenceIndicator(score=0.0, is_calibrated=False, method="heuristic", description="Doc number missing"),
            flags=["MISSING_DOCUMENT_NUMBER"],
            message="Document/Invoice number is missing",
        )

    cleaned = str(raw_value).strip()
    if not cleaned:
        return ExtractedField(
            name="document_number",
            raw_value=raw_value,
            normalized_value=None,
            status=FieldStatus.MISSING,
            confidence=ConfidenceIndicator(score=0.0, is_calibrated=False, method="heuristic", description="Doc number empty"),
            flags=["MISSING_DOCUMENT_NUMBER"],
            message="Document number is empty",
        )

    # Remove standard prefix labels like "INV-", "INVOICE #", "Receipt:"
    sanitized = re.sub(r"^(?:invoice|inv|receipt|bill|doc|no|num|#)[\s#:.-]*", "", cleaned, flags=re.IGNORECASE).strip()
    if not sanitized or not re.search(r"[a-zA-Z0-9]", sanitized):
        return ExtractedField(
            name="document_number",
            raw_value=raw_value,
            normalized_value=None,
            status=FieldStatus.UNREADABLE,
            confidence=ConfidenceIndicator(score=0.2, is_calibrated=False, method="heuristic", description="Unreadable doc number"),
            flags=["UNREADABLE_DOCUMENT_NUMBER"],
            message=f"Document number '{cleaned}' has no alphanumeric identifier",
        )

    return ExtractedField(
        name="document_number",
        raw_value=raw_value,
        normalized_value=sanitized,
        status=FieldStatus.VALID,
        confidence=ConfidenceIndicator(score=0.9, is_calibrated=False, method="heuristic", description="Standard alphanumeric ID"),
        flags=[],
        message=f"Document number recognized: {sanitized}",
    )


def validate_document_fields(
    raw_fields: Dict[str, Any],
    reference_date: Optional[date] = None,
    tolerance: Decimal = DEFAULT_TOLERANCE,
) -> DocumentValidationResult:
    """Run full deterministic validation on raw extracted document fields.

    Evaluates:
        - vendor
        - document_number
        - date (calendar and ambiguity check)
        - subtotal (Decimal)
        - tax (Decimal)
        - total (Decimal)
        - subtotal + tax == total arithmetic reconciliation
    """
    fields: Dict[str, ExtractedField] = {}
    overall_flags: List[str] = []

    # 1. Vendor
    v_field = validate_vendor(raw_fields.get("vendor"))
    fields["vendor"] = v_field
    overall_flags.extend(v_field.flags)

    # 2. Document Number
    d_field = validate_document_number(raw_fields.get("document_number"))
    fields["document_number"] = d_field
    overall_flags.extend(d_field.flags)

    # 3. Date
    raw_date = raw_fields.get("date")
    iso_date, date_status, date_flags, date_msg = parse_and_validate_date(raw_date, reference_date=reference_date)
    conf_score = 0.95 if date_status == FieldStatus.VALID else (0.5 if date_status == FieldStatus.AMBIGUOUS else 0.1)
    fields["date"] = ExtractedField(
        name="date",
        raw_value=raw_date,
        normalized_value=iso_date,
        status=date_status,
        confidence=ConfidenceIndicator(score=conf_score, is_calibrated=False, method="date_parser", description=date_msg),
        flags=date_flags,
        message=date_msg,
    )
    overall_flags.extend(date_flags)

    # 4. Monetary fields: subtotal, tax, total
    money_dec: Dict[str, Optional[Decimal]] = {}
    for m_name in ["subtotal", "tax", "total"]:
        raw_m = raw_fields.get(m_name)
        allow_neg = (m_name == "tax")  # Tax credits can sometimes be negative
        dec_val, m_status, m_flags, m_msg = parse_monetary_value(raw_m, allow_negative=allow_neg, field_name=m_name)
        money_dec[m_name] = dec_val
        m_conf = 0.95 if m_status == FieldStatus.VALID else 0.1
        fields[m_name] = ExtractedField(
            name=m_name,
            raw_value=str(raw_m) if raw_m is not None else None,
            normalized_value=str(dec_val) if dec_val is not None else None,
            status=m_status,
            confidence=ConfidenceIndicator(score=m_conf, is_calibrated=False, method="monetary_parser", description=m_msg),
            flags=m_flags,
            message=m_msg,
        )
        overall_flags.extend(m_flags)

    # 5. Arithmetic verification
    arithmetic_res = verify_arithmetic(
        subtotal=money_dec["subtotal"],
        tax=money_dec["tax"],
        total=money_dec["total"],
        tolerance=tolerance,
    )
    if not arithmetic_res.is_valid:
        if arithmetic_res.status == ArithmeticStatus.MISMATCH:
            overall_flags.append("ARITHMETIC_MISMATCH")
        elif arithmetic_res.status == ArithmeticStatus.INSUFFICIENT_DATA:
            overall_flags.append("ARITHMETIC_INCOMPLETE")

    # Document Validity: True only if no critical blocking flags exist
    blocking_statuses = {FieldStatus.MISSING, FieldStatus.UNREADABLE, FieldStatus.INCONSISTENT}
    critical_fields_fail = any(
        fields[k].status in blocking_statuses
        for k in ["vendor", "document_number", "date", "total"]
    )
    arithmetic_fail = (arithmetic_res.status == ArithmeticStatus.MISMATCH)
    is_valid = (not critical_fields_fail) and (not arithmetic_fail)

    summary_parts = []
    if is_valid:
        summary_parts.append("Document passed deterministic validation.")
    else:
        summary_parts.append("Document validation failed.")
    if arithmetic_res.status == ArithmeticStatus.BALANCED:
        summary_parts.append("Arithmetic balanced.")
    elif arithmetic_res.status == ArithmeticStatus.TOLERANCE_ACCEPTED:
        summary_parts.append("Arithmetic within rounding tolerance.")
    elif arithmetic_res.status == ArithmeticStatus.MISMATCH:
        summary_parts.append(f"Arithmetic mismatch: {arithmetic_res.message}.")

    if overall_flags:
        summary_parts.append(f"Flags: {', '.join(sorted(set(overall_flags)))}.")

    return DocumentValidationResult(
        is_valid=is_valid,
        fields=fields,
        arithmetic=arithmetic_res,
        overall_flags=sorted(list(set(overall_flags))),
        summary=" ".join(summary_parts),
    )
