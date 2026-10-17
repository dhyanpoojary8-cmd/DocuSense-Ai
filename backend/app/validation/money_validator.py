"""Deterministic monetary value parser and validator using Decimal arithmetic."""
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import List, Optional, Tuple

from app.models.field import FieldStatus


# Known currency symbols and codes
CURRENCY_PREFIXES = r"(?:[\$\€\£\¥\₹\₩\₽]|USD|EUR|GBP|CAD|AUD|INR|JPY|CHF|SGD|NZD|RM|kr|Rp|IDR)"
CURRENCY_PATTERN = re.compile(CURRENCY_PREFIXES, re.IGNORECASE)

# Currencies that traditionally operate on integer units without decimals
ZERO_DECIMAL_CURRENCIES = {"IDR", "JPY", "KRW", "VND", "CLP"}


def parse_monetary_value(
    raw_value: Optional[str],
    allow_negative: bool = False,
    field_name: str = "amount",
    currency_hint: Optional[str] = None,
    locale_hint: Optional[str] = None,
) -> Tuple[Optional[Decimal], FieldStatus, List[str], str]:
    """Parse a raw monetary string into a Decimal.

    Args:
        raw_value: Raw input text from OCR, PDF, or user.
        allow_negative: Whether negative amounts (e.g. tax credit) are permitted.
        field_name: Contextual field name for diagnostic logging.
        currency_hint: Optional ISO currency code (e.g. 'IDR', 'USD', 'EUR').
        locale_hint: Optional locale string (e.g. 'id_ID', 'en_US', 'de_DE').

    Returns:
        (normalized_decimal, status, flags, message)
    """
    if raw_value is None:
        return None, FieldStatus.MISSING, [f"MISSING_{field_name.upper()}"], f"{field_name.title()} is missing from document"

    cleaned = str(raw_value).strip()
    if not cleaned:
        return None, FieldStatus.MISSING, [f"MISSING_{field_name.upper()}"], f"{field_name.title()} is empty"

    # Auto-detect currency hint if Rp or IDR in text
    detected_currency = currency_hint.upper() if currency_hint else None
    if re.search(r"\b(?:Rp|IDR)\b", cleaned, re.IGNORECASE) or (locale_hint and "id" in locale_hint.lower()):
        detected_currency = "IDR"

    # Check for accounting negative notation: (123.45)
    is_negative = False
    if cleaned.startswith("(") and cleaned.endswith(")"):
        is_negative = True
        cleaned = cleaned[1:-1].strip()
    elif cleaned.startswith("-"):
        is_negative = True
        cleaned = cleaned[1:].strip()

    # Strip currency symbols and whitespace
    cleaned = CURRENCY_PATTERN.sub("", cleaned).strip()

    # If nothing left or no digits exist
    if not cleaned or not re.search(r"\d", cleaned):
        return None, FieldStatus.UNREADABLE, [f"UNREADABLE_{field_name.upper()}"], f"Could not extract numeric value from '{raw_value}'"

    # Check for unreadable characters
    cleaned_no_separators = re.sub(r"[\d\s\.,]", "", cleaned)
    if cleaned_no_separators:
        return None, FieldStatus.UNREADABLE, [f"UNREADABLE_{field_name.upper()}"], f"Value '{raw_value}' contains unreadable characters: {cleaned_no_separators}"

    # Remove internal spaces
    compact = re.sub(r"\s+", "", cleaned)

    # Special handling for zero-decimal currencies (e.g. IDR: "99,000" or "99.000" means integer 99000)
    if detected_currency in ZERO_DECIMAL_CURRENCIES:
        # In IDR / zero-decimal mode, flag amounts where a separator is followed by only 1-2 digits as AMBIGUOUS
        if re.search(r"[,.]\d{1,2}(?!\d)", compact):
            return (
                None,
                FieldStatus.AMBIGUOUS,
                [f"AMBIGUOUS_{field_name.upper()}"],
                f"Value '{raw_value}' in {detected_currency} has ambiguous separator followed by only 1-2 digits",
            )

        # Standard integer rupiah: remove all comma and dot grouping separators
        normalized_str = compact.replace(",", "").replace(".", "")

        try:
            val = Decimal(normalized_str)
        except (InvalidOperation, ValueError):
            return None, FieldStatus.UNREADABLE, [f"UNREADABLE_{field_name.upper()}"], f"Failed to parse Decimal from '{raw_value}'"

        if is_negative:
            val = -val
        quantized = val.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        return quantized, FieldStatus.VALID, [], f"Successfully parsed {field_name.title()} as {quantized} {detected_currency}"

    # Standard decimal currencies (USD, EUR, GBP, etc.)
    if "," in compact and "." in compact:
        last_comma = compact.rfind(",")
        last_dot = compact.rfind(".")
        if last_comma > last_dot:
            # European: 1.234,56
            normalized_str = compact.replace(".", "").replace(",", ".")
        else:
            # US/Standard: 1,234.56
            normalized_str = compact.replace(",", "")
    elif "," in compact:
        parts = compact.split(",")
        if len(parts) == 2 and len(parts[1]) in (1, 2):
            normalized_str = compact.replace(",", ".")
        elif len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            normalized_str = compact.replace(",", "")
        elif len(parts) == 2:
            normalized_str = compact.replace(",", ".")
        else:
            return None, FieldStatus.UNREADABLE, [f"AMBIGUOUS_NUMBER_FORMAT_{field_name.upper()}"], f"Unusual separator pattern in '{raw_value}'"
    elif "." in compact:
        parts = compact.split(".")
        if len(parts) > 2:
            normalized_str = compact.replace(".", "")
        else:
            normalized_str = compact
    else:
        normalized_str = compact

    try:
        val = Decimal(normalized_str)
    except (InvalidOperation, ValueError):
        return None, FieldStatus.UNREADABLE, [f"UNREADABLE_{field_name.upper()}"], f"Failed to parse Decimal from '{raw_value}'"

    if is_negative:
        val = -val

    if val < Decimal("0.00") and not allow_negative:
        flags = [f"NEGATIVE_{field_name.upper()}"]
        quantized = val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return quantized, FieldStatus.INCONSISTENT, flags, f"{field_name.title()} cannot be negative ({quantized})"

    quantized = val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return quantized, FieldStatus.VALID, [], f"Successfully parsed {field_name.title()} as {quantized}"
