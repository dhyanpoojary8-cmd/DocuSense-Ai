"""Unit and integration tests for deterministic validation engine.

Covers:
- Monetary parsing (Decimal precision, currency symbols, US/EU formats, negatives, garbage text)
- Date validation (ISO, US, EU, text month, leap year, Feb 30, ambiguous dates, future dates)
- Arithmetic reconciliation (exact balance, rounding tolerance, mismatches, missing values)
- Document-level validation & anomaly flag generation
- Synthetic fixtures for edge cases
"""
from datetime import date
from decimal import Decimal
import pytest

from app.models.field import FieldStatus
from app.models.validation import ArithmeticStatus
from app.validation.arithmetic import verify_arithmetic
from app.validation.date_validator import parse_and_validate_date
from app.validation.money_validator import parse_monetary_value
from app.validation.rules_engine import (
    validate_document_fields,
    validate_vendor,
    validate_document_number,
)


# ============================================================================
# 1. MONETARY PARSER TESTS
# ============================================================================

class TestMonetaryParser:
    """Test suite for Decimal monetary parsing."""

    def test_standard_us_currency(self):
        val, status, flags, msg = parse_monetary_value("$1,234.56", field_name="total")
        assert status == FieldStatus.VALID
        assert val == Decimal("1234.56")
        assert flags == []

    def test_european_currency_format(self):
        val, status, flags, msg = parse_monetary_value("€ 1.234,56", field_name="total")
        assert status == FieldStatus.VALID
        assert val == Decimal("1234.56")
        assert flags == []

    def test_comma_decimal_simple(self):
        val, status, flags, msg = parse_monetary_value("12,50", field_name="tax")
        assert status == FieldStatus.VALID
        assert val == Decimal("12.50")
        assert flags == []

    def test_plain_integer(self):
        val, status, flags, msg = parse_monetary_value("500", field_name="subtotal")
        assert status == FieldStatus.VALID
        assert val == Decimal("500.00")

    def test_whitespace_inside_number(self):
        val, status, flags, msg = parse_monetary_value("1 500.25", field_name="total")
        assert status == FieldStatus.VALID
        assert val == Decimal("1500.25")

    def test_negative_parentheses(self):
        val, status, flags, msg = parse_monetary_value("($45.50)", allow_negative=True, field_name="tax")
        assert status == FieldStatus.VALID
        assert val == Decimal("-45.50")

    def test_negative_disallowed(self):
        val, status, flags, msg = parse_monetary_value("-15.00", allow_negative=False, field_name="total")
        assert status == FieldStatus.INCONSISTENT
        assert "NEGATIVE_TOTAL" in flags
        assert val == Decimal("-15.00")

    def test_missing_monetary_value(self):
        val, status, flags, msg = parse_monetary_value(None, field_name="subtotal")
        assert status == FieldStatus.MISSING
        assert val is None
        assert "MISSING_SUBTOTAL" in flags

    def test_empty_monetary_value(self):
        val, status, flags, msg = parse_monetary_value("   ", field_name="tax")
        assert status == FieldStatus.MISSING
        assert val is None
        assert "MISSING_TAX" in flags

    def test_unreadable_garbage_characters(self):
        val, status, flags, msg = parse_monetary_value("$12.a5", field_name="total")
        assert status == FieldStatus.UNREADABLE
        assert val is None
        assert "UNREADABLE_TOTAL" in flags

    def test_unreadable_no_digits(self):
        val, status, flags, msg = parse_monetary_value("N/A", field_name="total")
        assert status == FieldStatus.UNREADABLE
        assert val is None
        assert "UNREADABLE_TOTAL" in flags


# ============================================================================
# 2. DATE PARSER & VALIDATOR TESTS
# ============================================================================

class TestDateValidator:
    """Test suite for date parsing, calendar validation, and ambiguity detection."""

    def test_iso_date_valid(self):
        iso_str, status, flags, msg = parse_and_validate_date("2024-04-15")
        assert status == FieldStatus.VALID
        assert iso_str == "2024-04-15"
        assert flags == []

    def test_text_month_format(self):
        iso_str, status, flags, msg = parse_and_validate_date("15 Oct 2024")
        assert status == FieldStatus.VALID
        assert iso_str == "2024-10-15"

    def test_unambiguous_day_greater_than_12(self):
        # 25 is day because > 12
        iso_str, status, flags, msg = parse_and_validate_date("25/08/2024")
        assert status == FieldStatus.VALID
        assert iso_str == "2024-08-25"

    def test_unambiguous_month_greater_than_12(self):
        # 16 is day because > 12
        iso_str, status, flags, msg = parse_and_validate_date("08/16/2024")
        assert status == FieldStatus.VALID
        assert iso_str == "2024-08-16"

    def test_ambiguous_date_both_under_12(self):
        # 04/05/2024 could be April 5th or May 4th
        iso_str, status, flags, msg = parse_and_validate_date("04/05/2024")
        assert status == FieldStatus.AMBIGUOUS
        assert "AMBIGUOUS_DATE_FORMAT" in flags
        assert "ambiguous" in msg.lower()

    def test_same_day_and_month_not_ambiguous(self):
        # 05/05/2024 has day == month, so no ambiguity
        iso_str, status, flags, msg = parse_and_validate_date("05/05/2024")
        assert status == FieldStatus.VALID
        assert iso_str == "2024-05-05"

    def test_leap_year_valid(self):
        # 2024 is a leap year
        iso_str, status, flags, msg = parse_and_validate_date("2024-02-29")
        assert status == FieldStatus.VALID
        assert iso_str == "2024-02-29"

    def test_non_leap_year_feb_29_invalid(self):
        # 2023 is not a leap year
        iso_str, status, flags, msg = parse_and_validate_date("2023-02-29")
        assert status == FieldStatus.INCONSISTENT
        assert "INVALID_CALENDAR_DATE" in flags
        assert iso_str is None

    def test_february_30_invalid(self):
        iso_str, status, flags, msg = parse_and_validate_date("2024-02-30")
        assert status == FieldStatus.INCONSISTENT
        assert "INVALID_CALENDAR_DATE" in flags

    def test_month_13_invalid(self):
        iso_str, status, flags, msg = parse_and_validate_date("13/15/2024")
        assert status == FieldStatus.INCONSISTENT
        assert "INVALID_CALENDAR_DATE" in flags

    def test_suspicious_future_date(self):
        ref = date(2024, 1, 1)
        iso_str, status, flags, msg = parse_and_validate_date("2025-06-01", reference_date=ref)
        assert status == FieldStatus.INCONSISTENT
        assert "SUSPICIOUS_FUTURE_DATE" in flags

    def test_missing_date(self):
        iso_str, status, flags, msg = parse_and_validate_date(None)
        assert status == FieldStatus.MISSING
        assert iso_str is None
        assert "MISSING_DATE" in flags

    def test_unreadable_date(self):
        iso_str, status, flags, msg = parse_and_validate_date("sometime last week")
        assert status == FieldStatus.UNREADABLE
        assert iso_str is None
        assert "UNREADABLE_DATE" in flags


# ============================================================================
# 3. ARITHMETIC RECONCILIATION TESTS
# ============================================================================

class TestArithmeticVerification:
    """Test suite for subtotal + tax == total reconciliation using Decimal."""

    def test_exact_arithmetic_balance(self):
        subtotal = Decimal("100.00")
        tax = Decimal("8.25")
        total = Decimal("108.25")
        res = verify_arithmetic(subtotal, tax, total)
        assert res.is_valid is True
        assert res.status == ArithmeticStatus.BALANCED
        assert res.difference == "0.00"

    def test_tolerance_accepted_rounding(self):
        # Subtotal: $10.00, Tax: $0.83, Extracted Total: $10.82 (diff = $0.01)
        subtotal = Decimal("10.00")
        tax = Decimal("0.83")
        total = Decimal("10.82")
        res = verify_arithmetic(subtotal, tax, total, tolerance=Decimal("0.02"))
        assert res.is_valid is True
        assert res.status == ArithmeticStatus.TOLERANCE_ACCEPTED
        assert res.difference == "0.01"

    def test_arithmetic_mismatch_exceeds_tolerance(self):
        subtotal = Decimal("100.00")
        tax = Decimal("10.00")
        total = Decimal("115.00")  # Difference $5.00 > $0.02
        res = verify_arithmetic(subtotal, tax, total)
        assert res.is_valid is False
        assert res.status == ArithmeticStatus.MISMATCH
        assert res.difference == "5.00"

    def test_missing_tax_when_subtotal_equals_total(self):
        # Zero tax itemized
        subtotal = Decimal("75.00")
        tax = None
        total = Decimal("75.00")
        res = verify_arithmetic(subtotal, tax, total)
        assert res.is_valid is True
        assert res.status == ArithmeticStatus.BALANCED

    def test_missing_tax_when_subtotal_less_than_total(self):
        subtotal = Decimal("50.00")
        tax = None
        total = Decimal("55.00")
        res = verify_arithmetic(subtotal, tax, total)
        assert res.is_valid is False
        assert res.status == ArithmeticStatus.INSUFFICIENT_DATA
        assert "implied tax" in res.message

    def test_missing_subtotal_returns_insufficient_data(self):
        res = verify_arithmetic(None, Decimal("5.00"), Decimal("55.00"))
        assert res.is_valid is False
        assert res.status == ArithmeticStatus.INSUFFICIENT_DATA
        assert "subtotal" in res.message

    def test_missing_total_returns_insufficient_data(self):
        res = verify_arithmetic(Decimal("50.00"), Decimal("5.00"), None)
        assert res.is_valid is False
        assert res.status == ArithmeticStatus.INSUFFICIENT_DATA
        assert "total" in res.message


# ============================================================================
# 4. FIELD VALIDATORS (VENDOR & DOC NUMBER)
# ============================================================================

class TestFieldValidators:
    """Test suite for vendor and document number rules."""

    def test_valid_vendor(self):
        f = validate_vendor("acme corp services")
        assert f.status == FieldStatus.VALID
        assert f.normalized_value == "Acme Corp Services"

    def test_missing_vendor(self):
        f = validate_vendor(None)
        assert f.status == FieldStatus.MISSING
        assert f.normalized_value is None
        assert "MISSING_VENDOR" in f.flags

    def test_unreadable_vendor_gibberish(self):
        f = validate_vendor("###$$$@#")
        assert f.status == FieldStatus.UNREADABLE
        assert f.normalized_value is None

    def test_valid_document_number(self):
        f = validate_document_number("INV-2024-00123")
        assert f.status == FieldStatus.VALID
        assert f.normalized_value == "2024-00123"

    def test_missing_document_number(self):
        f = validate_document_number(None)
        assert f.status == FieldStatus.MISSING
        assert "MISSING_DOCUMENT_NUMBER" in f.flags


# ============================================================================
# 5. FULL DOCUMENT VALIDATION & SYNTHETIC FIXTURES
# ============================================================================

class TestDocumentValidationEngine:
    """Integration test suite for the complete document validation engine."""

    def test_perfect_document_passes(self):
        raw = {
            "vendor": "Acme Technologies",
            "document_number": "INV-9988",
            "date": "2024-05-18",
            "subtotal": "$100.00",
            "tax": "$8.50",
            "total": "$108.50",
        }
        res = validate_document_fields(raw)
        assert res.is_valid is True
        assert res.arithmetic.status == ArithmeticStatus.BALANCED
        assert res.overall_flags == []

    def test_synthetic_fixture_mismatched_totals(self):
        """Fixture: Tax + subtotal does not match total."""
        raw = {
            "vendor": "Corner Store",
            "document_number": "REC-101",
            "date": "2024-06-01",
            "subtotal": "50.00",
            "tax": "5.00",
            "total": "60.00",  # Mismatch: 50 + 5 != 60
        }
        res = validate_document_fields(raw)
        assert res.is_valid is False
        assert res.arithmetic.status == ArithmeticStatus.MISMATCH
        assert "ARITHMETIC_MISMATCH" in res.overall_flags

    def test_synthetic_fixture_missing_vendor_and_tax(self):
        """Fixture: Vendor and tax are missing."""
        raw = {
            "vendor": None,
            "document_number": "DOC-777",
            "date": "2024-03-12",
            "subtotal": "100.00",
            "tax": None,
            "total": "100.00",
        }
        res = validate_document_fields(raw)
        assert res.is_valid is False  # Missing vendor is critical
        assert "MISSING_VENDOR" in res.overall_flags
        assert res.fields["vendor"].status == FieldStatus.MISSING
        assert res.fields["tax"].status == FieldStatus.MISSING
        # Arithmetic should be balanced since subtotal == total with zero tax
        assert res.arithmetic.status == ArithmeticStatus.BALANCED

    def test_synthetic_fixture_unreadable_text(self):
        """Fixture: OCR garbage for numeric and date values."""
        raw = {
            "vendor": "###",
            "document_number": "???",
            "date": "last-monday",
            "subtotal": "XX.YY",
            "tax": "N/A",
            "total": "100.00",
        }
        res = validate_document_fields(raw)
        assert res.is_valid is False
        assert res.fields["vendor"].status == FieldStatus.UNREADABLE
        assert res.fields["date"].status == FieldStatus.UNREADABLE
        assert res.fields["subtotal"].status == FieldStatus.UNREADABLE
        assert res.arithmetic.status == ArithmeticStatus.INSUFFICIENT_DATA

    def test_synthetic_fixture_ambiguous_date(self):
        """Fixture: Date 03/04/2024 is ambiguous."""
        raw = {
            "vendor": "Global Logistics",
            "document_number": "GL-554",
            "date": "03/04/2024",
            "subtotal": "200.00",
            "tax": "20.00",
            "total": "220.00",
        }
        res = validate_document_fields(raw)
        assert res.fields["date"].status == FieldStatus.AMBIGUOUS
        assert "AMBIGUOUS_DATE_FORMAT" in res.overall_flags
        assert res.arithmetic.status == ArithmeticStatus.BALANCED
