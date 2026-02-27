"""
Models for validation results and quality scoring.

Supports multi-dimensional scoring across:
- Code Quality, Standards Adherence, Performance,
  Parameterization, Verbosity, Data Accuracy
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
import uuid


class ValidationCategory(str, Enum):
    """Categories of validation checks."""
    # Original categories (backward-compatible)
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    LOGIC = "logic"
    ERROR_HANDLING = "error_handling"
    # Multi-dimensional categories
    CODE_QUALITY = "code_quality"
    STANDARDS = "standards"
    PERFORMANCE = "performance"
    PARAMETERIZATION = "parameterization"
    VERBOSITY = "verbosity"
    DATA_ACCURACY = "data_accuracy"


class ValidationSeverity(str, Enum):
    """Severity level of validation results."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationResult(BaseModel):
    """Result of a single validation check."""
    validation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str
    check_name: str
    passed: bool
    expected: Optional[str] = None
    actual: Optional[str] = None
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    category: ValidationCategory
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class ScoreDimension(BaseModel):
    """Score for a single quality dimension."""
    name: str
    score: float = Field(..., ge=0.0, le=100.0)
    weight: float = Field(..., ge=0.0, le=1.0)
    description: str = ""
    checks: List[Dict[str, Any]] = Field(default_factory=list)
    passed_count: int = 0
    failed_count: int = 0

    @property
    def grade(self) -> str:
        if self.score >= 90:
            return "A"
        elif self.score >= 80:
            return "B"
        elif self.score >= 70:
            return "C"
        elif self.score >= 60:
            return "D"
        return "F"


class QualityScore(BaseModel):
    """Quality score for a conversion job."""
    job_id: str
    overall_score: float = Field(..., ge=0.0, le=1.0)
    categories: Dict[str, float] = Field(default_factory=dict)
    dimensions: List[ScoreDimension] = Field(default_factory=list)
    validation_count: int = 0
    passed_count: int = 0
    failed_count: int = 0
    metrics: Dict[str, Any] = Field(default_factory=dict)
    calculated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def grade(self) -> str:
        """Get letter grade based on overall score."""
        pct = self.overall_score * 100
        if pct >= 90:
            return "A"
        elif pct >= 80:
            return "B"
        elif pct >= 70:
            return "C"
        elif pct >= 60:
            return "D"
        else:
            return "F"

    @property
    def is_promotable(self) -> bool:
        """Check if score meets minimum threshold for promotion (70%)."""
        return self.overall_score >= 0.70


class ValidationReport(BaseModel):
    """Complete validation report for a job."""
    job_id: str
    job_name: str
    score: QualityScore
    results: List[ValidationResult]
    source_metrics: Dict[str, Any] = Field(default_factory=dict)
    output_metrics: Dict[str, Any] = Field(default_factory=dict)
    comparison_summary: Optional[str] = None
    gate_results: Optional[Dict[str, Any]] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class ValidationRunRequest(BaseModel):
    """Request to run validation on a job."""
    job_id: str
    compare_with_source: bool = True
    source_table: Optional[str] = None
    output_table: Optional[str] = None
