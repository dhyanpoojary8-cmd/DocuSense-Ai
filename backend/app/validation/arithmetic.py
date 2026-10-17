"""Deterministic arithmetic reconciler verifying subtotal + tax == total using Decimal."""
from decimal import Decimal
from typing import Optional

from app.models.validation import ArithmeticStatus, ArithmeticVerificationResult


DEFAULT_DECIMAL_TOLERANCE = Decimal("0.02")
DEFAULT_ZERO_DECIMAL_TOLERANCE = Decimal("100.00")  # Tolerance for IDR/JPY where small roundings/adjustments happen in hundreds
DEFAULT_TOLERANCE = DEFAULT_DECIMAL_TOLERANCE


def verify_arithmetic(
    subtotal: Optional[Decimal],
    tax: Optional[Decimal],
    total: Optional[Decimal],
    tolerance: Optional[Decimal] = None,
    currency: Optional[str] = None,
) -> ArithmeticVerificationResult:
    """Verify whether subtotal + tax matches total within the currency-aware tolerance.

    Rules:
        - Never invent missing values.
        - Uses Python Decimal for exact arithmetic.
        - Adjusts default tolerance based on currency (e.g. 0.02 for USD/EUR, 100 for IDR).
    """
    curr_upper = currency.upper() if currency else "USD"
    if tolerance is None:
        if curr_upper in {"IDR", "JPY", "KRW"}:
            effective_tolerance = DEFAULT_ZERO_DECIMAL_TOLERANCE
        else:
            effective_tolerance = DEFAULT_DECIMAL_TOLERANCE
    else:
        effective_tolerance = tolerance

    subtotal_str = str(subtotal) if subtotal is not None else None
    tax_str = str(tax) if tax is not None else None
    total_str = str(total) if total is not None else None
    tol_str = str(effective_tolerance)

    # Case 1: Total or subtotal completely missing
    if total is None or subtotal is None:
        missing_fields = []
        if subtotal is None:
            missing_fields.append("subtotal")
        if total is None:
            missing_fields.append("total")
        return ArithmeticVerificationResult(
            status=ArithmeticStatus.INSUFFICIENT_DATA,
            subtotal=subtotal_str,
            tax=tax_str,
            calculated_total=None,
            extracted_total=total_str,
            difference=None,
            tolerance=tol_str,
            is_valid=False,
            message=f"Insufficient arithmetic inputs: missing {', '.join(missing_fields)}",
        )

    # Case 2: Tax is missing
    if tax is None:
        if subtotal == total:
            return ArithmeticVerificationResult(
                status=ArithmeticStatus.BALANCED,
                subtotal=subtotal_str,
                tax=None,
                calculated_total=subtotal_str,
                extracted_total=total_str,
                difference="0.00",
                tolerance=tol_str,
                is_valid=True,
                message="Subtotal matches total exactly; document has no separate tax item",
            )
        if subtotal < total:
            implied_tax = (total - subtotal)
            return ArithmeticVerificationResult(
                status=ArithmeticStatus.INSUFFICIENT_DATA,
                subtotal=subtotal_str,
                tax=None,
                calculated_total=None,
                extracted_total=total_str,
                difference=str(implied_tax),
                tolerance=tol_str,
                is_valid=False,
                message=f"Tax was not extracted from document (implied tax would be {implied_tax})",
            )
        # subtotal > total
        excess = (subtotal - total)
        return ArithmeticVerificationResult(
            status=ArithmeticStatus.MISMATCH,
            subtotal=subtotal_str,
            tax=None,
            calculated_total=subtotal_str,
            extracted_total=total_str,
            difference=str(excess),
            tolerance=tol_str,
            is_valid=False,
            message=f"Subtotal ({subtotal}) exceeds total ({total}) by {excess}",
        )

    # Case 3: All three values are available
    calculated = subtotal + tax
    diff = abs(calculated - total)
    calc_str = str(calculated)
    diff_str = str(diff)

    if diff == Decimal("0.00") or diff == Decimal("0"):
        return ArithmeticVerificationResult(
            status=ArithmeticStatus.BALANCED,
            subtotal=subtotal_str,
            tax=tax_str,
            calculated_total=calc_str,
            extracted_total=total_str,
            difference=diff_str,
            tolerance=tol_str,
            is_valid=True,
            message=f"Arithmetic balanced: {subtotal} + {tax} == {total}",
        )

    if diff <= effective_tolerance:
        return ArithmeticVerificationResult(
            status=ArithmeticStatus.TOLERANCE_ACCEPTED,
            subtotal=subtotal_str,
            tax=tax_str,
            calculated_total=calc_str,
            extracted_total=total_str,
            difference=diff_str,
            tolerance=tol_str,
            is_valid=True,
            message=f"Arithmetic matches within tolerance ({diff} <= {effective_tolerance} {curr_upper}): {subtotal} + {tax} ≈ {total}",
        )

    # Mismatch exceeds tolerance
    return ArithmeticVerificationResult(
        status=ArithmeticStatus.MISMATCH,
        subtotal=subtotal_str,
        tax=tax_str,
        calculated_total=calc_str,
        extracted_total=total_str,
        difference=diff_str,
        tolerance=tol_str,
        is_valid=False,
        message=f"Arithmetic mismatch: subtotal ({subtotal}) + tax ({tax}) = {calculated}, but extracted total is {total} (difference of {diff})",
    )
