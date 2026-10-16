"""Domain models for deterministic document validation results."""
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from app.models.field import ExtractedField


class ArithmeticStatus(str, Enum):
    """Result status of arithmetic reconciliation."""
    BALANCED = "BALANCED"
    MISMATCH = "MISMATCH"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    TOLERANCE_ACCEPTED = "TOLERANCE_ACCEPTED"


class ArithmeticVerificationResult(BaseModel):
    """Detailed result of verifying subtotal + tax == total."""
    status: ArithmeticStatus
    subtotal: Optional[str] = Field(default=None, description="Normalized subtotal formatted as string")
    tax: Optional[str] = Field(default=None, description="Normalized tax formatted as string")
    calculated_total: Optional[str] = Field(default=None, description="Computed subtotal + tax")
    extracted_total: Optional[str] = Field(default=None, description="Extracted document total")
    difference: Optional[str] = Field(default=None, description="Abs difference between calculated and extracted")
    tolerance: str = Field(default="0.02", description="Permitted rounding tolerance threshold in currency units")
    is_valid: bool = Field(description="True if balanced or within tolerance, False otherwise")
    message: str = Field(description="Human-readable explanation of arithmetic result")


class DocumentValidationResult(BaseModel):
    """Overall validation outcome combining field checks and arithmetic verification."""
    is_valid: bool = Field(description="True if document passes all critical checks without blocking errors")
    fields: Dict[str, ExtractedField] = Field(description="Map of field name to ExtractedField details")
    arithmetic: ArithmeticVerificationResult = Field(description="Arithmetic reconciliation check details")
    overall_flags: List[str] = Field(default_factory=list, description="Aggregated diagnostic flags across all fields")
    summary: str = Field(description="Brief summary of document verification state")
