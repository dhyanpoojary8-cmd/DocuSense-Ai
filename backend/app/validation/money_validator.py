"""Deterministic monetary value parser and validator using Decimal arithmetic."""
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import List, Optional, Tuple

from app.models.field import FieldStatus


# Known currency symbols and codes to strip cleanly
CURRENCY_PREFIXES = r"(?:[\$\€\£\¥\₹\₩\₽]|USD|EUR|GBP|CAD|AUD|INR|JPY|CHF|SGD|NZD|RM|kr)"
CURRENCY_PATTERN = re.compile(CURRENCY_PREFIXES, re.IGNORECASE)


def parse_monetary_value(
    raw_value: Optional[str],
    allow_negative: bool = False,
    field_name: str = "amount",
) -> Tuple[Optional[Decimal], FieldStatus, List[str], str]:
    """Parse a raw monetary string into a Decimal with 2 decimal places.

    Returns:
        (normalized_decimal, status, flags, message)

    Rules:
        - If input is empty/None -> MISSING (never invent a value)
        - If input contains garbled/non-numeric characters -> UNREADABLE
        - Correctly parses both US (1,234.56) and European (1.234,56) number conventions
        - Checks for negative amounts: (12.34) or -12.34
        - Quantizes to 2 decimal places (ROUND_HALF_UP)
    """
    if raw_value is None:
        return None, FieldStatus.MISSING, [f"MISSING_{field_name.upper()}"], f"{field_name.title()} is missing from document"

    cleaned = str(raw_value).strip()
    if not cleaned:
        return None, FieldStatus.MISSING, [f"MISSING_{field_name.upper()}"], f"{field_name.title()} is empty"

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

    # If nothing left or invalid characters exist
    if not cleaned or not re.search(r"\d", cleaned):
        return None, FieldStatus.UNREADABLE, [f"UNREADABLE_{field_name.upper()}"], f"Could not extract numeric value from '{raw_value}'"

    # Check for unreadable/garbage characters (only digits, comma, period, spaces allowed)
    cleaned_no_separators = re.sub(r"[\d\s\.,]", "", cleaned)
    if cleaned_no_separators:
        return None, FieldStatus.UNREADABLE, [f"UNREADABLE_{field_name.upper()}"], f"Value '{raw_value}' contains unreadable characters: {cleaned_no_separators}"

    # Remove internal spaces (e.g. "1 234,56" or "10 . 50")
    compact = re.sub(r"\s+", "", cleaned)

    # Detect comma vs period convention:
    # Case 1: Both comma and period present
    if "," in compact and "." in compact:
        last_comma = compact.rfind(",")
        last_dot = compact.rfind(".")
        if last_comma > last_dot:
            # European: 1.234,56 -> remove dots, replace comma with dot
            normalized_str = compact.replace(".", "").replace(",", ".")
        else:
            # US/Standard: 1,234.56 -> remove commas
            normalized_str = compact.replace(",", "")
    elif "," in compact:
        # Only comma present
        parts = compact.split(",")
        if len(parts) == 2 and len(parts[1]) in (1, 2):
            # Decimal comma: e.g. "12,50" -> "12.50"
            normalized_str = compact.replace(",", ".")
        elif len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            # Thousands separator: e.g. "1,234" or "1,234,567"
            normalized_str = compact.replace(",", "")
        elif len(parts) == 2:
            # Ambiguous separator (e.g. "12,345" where fractional has 3 digits)
            # Typically European 3-decimal fraction or thousands separator
            normalized_str = compact.replace(",", ".")
        else:
            return None, FieldStatus.UNREADABLE, [f"AMBIGUOUS_NUMBER_FORMAT_{field_name.upper()}"], f"Unusual separator pattern in '{raw_value}'"
    elif "." in compact:
        parts = compact.split(".")
        if len(parts) > 2:
            # Multiple periods without commas: e.g. "1.234.567"
            normalized_str = compact.replace(".", "")
        else:
            normalized_str = compact
    else:
        # Plain integer string
        normalized_str = compact

    # Convert to Decimal
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
