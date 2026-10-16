"""Deterministic date parser and calendar validator with ambiguity detection."""
import calendar
import re
from datetime import date, datetime
from typing import List, Optional, Tuple

from app.models.field import FieldStatus


# Month name mapping for textual dates
MONTH_NAMES = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9, "sept": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def parse_and_validate_date(
    raw_value: Optional[str],
    reference_date: Optional[date] = None,
    allow_future: bool = False,
    field_name: str = "date",
) -> Tuple[Optional[str], FieldStatus, List[str], str]:
    """Parse a date string into ISO format (YYYY-MM-DD) and validate calendar integrity.

    Returns:
        (iso_date_string, status, flags, message)

    Rules:
        - If input is empty/None -> MISSING (never invent a replacement)
        - If text is unreadable or gibberish -> UNREADABLE
        - If numeric day/month are both <= 12 (e.g. 04/05/2024) -> AMBIGUOUS
        - If calendar date is invalid (e.g. Feb 30, non-leap Feb 29) -> INCONSISTENT
        - If date is suspiciously far in future -> INCONSISTENT / SUSPICIOUS_FUTURE_DATE
    """
    if raw_value is None:
        return None, FieldStatus.MISSING, [f"MISSING_{field_name.upper()}"], f"{field_name.title()} is missing from document"

    cleaned = str(raw_value).strip()
    if not cleaned:
        return None, FieldStatus.MISSING, [f"MISSING_{field_name.upper()}"], f"{field_name.title()} is empty"

    if reference_date is None:
        reference_date = date.today()

    # Step 1: Text month check (e.g. "15 Oct 2024", "October 15, 2024", "15-Oct-2024")
    text_month_match = re.search(
        r"\b(\d{1,2})[-/\s]+([a-zA-Z]{3,9})[-/\s,]+(\d{2,4})\b|\b([a-zA-Z]{3,9})[-/\s]+(\d{1,2})[-/\s,]+(\d{2,4})\b",
        cleaned,
    )
    if text_month_match:
        groups = text_month_match.groups()
        if groups[0] is not None:
            day_str, month_str, year_str = groups[0], groups[1], groups[2]
        else:
            month_str, day_str, year_str = groups[3], groups[4], groups[5]

        month_key = month_str.lower()
        if month_key in MONTH_NAMES:
            month = MONTH_NAMES[month_key]
            day = int(day_str)
            year = int(year_str)
            if year < 100:
                year += 2000 if year < 70 else 1900
            return _check_calendar_and_plausibility(year, month, day, cleaned, reference_date, allow_future, field_name, is_ambiguous=False)

    # Step 2: ISO Date pattern (YYYY-MM-DD or YYYY/MM/DD or YYYY.MM.DD)
    iso_match = re.match(r"^(\d{4})[-/\.](\d{1,2})[-/\.](\d{1,2})$", cleaned)
    if iso_match:
        year = int(iso_match.group(1))
        month = int(iso_match.group(2))
        day = int(iso_match.group(3))
        return _check_calendar_and_plausibility(year, month, day, cleaned, reference_date, allow_future, field_name, is_ambiguous=False)

    # Step 3: Numeric Day/Month/Year or Month/Day/Year pattern (e.g. 15/04/2024, 04/15/2024, 04/05/2024)
    slash_match = re.match(r"^(\d{1,2})[-/\.](\d{1,2})[-/\.](\d{2,4})$", cleaned)
    if slash_match:
        part1 = int(slash_match.group(1))
        part2 = int(slash_match.group(2))
        year = int(slash_match.group(3))
        if year < 100:
            year += 2000 if year < 70 else 1900

        # Case A: Ambiguous (both part1 and part2 are <= 12)
        if 1 <= part1 <= 12 and 1 <= part2 <= 12 and part1 != part2:
            # We cannot deterministically know if part1 is month or day without locale
            # Default to US MM/DD representation for normalized string, but flag as AMBIGUOUS
            iso_us = f"{year:04d}-{part1:02d}-{part2:02d}"
            iso_eu = f"{year:04d}-{part2:02d}-{part1:02d}"
            flags = ["AMBIGUOUS_DATE_FORMAT"]
            msg = f"Date '{cleaned}' is ambiguous (could be {iso_us} or {iso_eu})"
            return iso_us, FieldStatus.AMBIGUOUS, flags, msg

        # Case B: Unambiguous Day > 12 -> must be DD/MM/YYYY
        if part1 > 12 and 1 <= part2 <= 12:
            return _check_calendar_and_plausibility(year, part2, part1, cleaned, reference_date, allow_future, field_name, is_ambiguous=False)

        # Case C: Unambiguous Month > 12 -> must be MM/DD/YYYY
        if part2 > 12 and 1 <= part1 <= 12:
            return _check_calendar_and_plausibility(year, part1, part2, cleaned, reference_date, allow_future, field_name, is_ambiguous=False)

        # Case D: Both parts equal (e.g. 05/05/2024)
        if part1 == part2 and 1 <= part1 <= 12:
            return _check_calendar_and_plausibility(year, part1, part2, cleaned, reference_date, allow_future, field_name, is_ambiguous=False)

        # Both parts > 12 or invalid
        return None, FieldStatus.INCONSISTENT, ["INVALID_CALENDAR_DATE"], f"Date '{cleaned}' contains invalid month/day values ({part1}, {part2})"

    # Unrecognized or noisy string
    return None, FieldStatus.UNREADABLE, [f"UNREADABLE_{field_name.upper()}"], f"Unrecognized date format in '{raw_value}'"


def _check_calendar_and_plausibility(
    year: int,
    month: int,
    day: int,
    raw_str: str,
    reference_date: date,
    allow_future: bool,
    field_name: str,
    is_ambiguous: bool,
) -> Tuple[Optional[str], FieldStatus, List[str], str]:
    """Validate calendar existence (month range, days in month, leap year) and chronological plausibility."""
    if not (1 <= month <= 12):
        return None, FieldStatus.INCONSISTENT, ["INVALID_CALENDAR_DATE"], f"Month {month} is out of range (1-12) in '{raw_str}'"

    # Days in month check (including leap year via calendar.monthrange)
    _, max_days = calendar.monthrange(year, month)
    if not (1 <= day <= max_days):
        leap_note = " (not a leap year)" if month == 2 and not calendar.isleap(year) else ""
        return None, FieldStatus.INCONSISTENT, ["INVALID_CALENDAR_DATE"], f"Day {day} does not exist in month {month}/{year}{leap_note}"

    try:
        parsed_date = date(year, month, day)
    except ValueError as e:
        return None, FieldStatus.INCONSISTENT, ["INVALID_CALENDAR_DATE"], f"Invalid date: {e}"

    iso_str = parsed_date.isoformat()
    flags: List[str] = []

    # Plausibility checks
    if parsed_date.year < 1980:
        flags.append("SUSPICIOUS_HISTORICAL_DATE")

    # Future date check (invoices rarely dated > 30 days in the future unless forward-dated contract)
    days_in_future = (parsed_date - reference_date).days
    if days_in_future > 30 and not allow_future:
        flags.append("SUSPICIOUS_FUTURE_DATE")
        return iso_str, FieldStatus.INCONSISTENT, flags, f"Date {iso_str} is {days_in_future} days in the future"

    status = FieldStatus.AMBIGUOUS if is_ambiguous else FieldStatus.VALID
    msg = f"Valid date: {iso_str}" if not flags else f"Date {iso_str} with warnings: {', '.join(flags)}"
    return iso_str, status, flags, msg
