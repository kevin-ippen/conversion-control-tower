"""
Models for conversion jobs and promotion workflow.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
import uuid


class ConversionStatus(str, Enum):
    """Status of a conversion job."""
    PENDING = "pending"
    PARSING = "parsing"
    CONVERTING = "converting"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SourceType(str, Enum):
    """Type of source file being converted."""
    SSIS = "ssis"
    STORED_PROC = "stored_proc"
    SQL_SCRIPT = "sql_script"
    INFORMATICA_PC = "informatica_pc"


class OutputFormat(str, Enum):
    """Target output format for conversion."""
    PYSPARK = "pyspark"
    DLT_SDP = "dlt_sdp"
    DBT = "dbt"


class SourceLocation(str, Enum):
    """Where the source code is located."""
    UPLOAD = "upload"
    WORKSPACE = "workspace"
    REPO = "repo"
    UC_VOLUME = "uc_volume"


class ValidationSource(str, Enum):
    """Source of validation/comparison data."""
    UC_TABLE = "uc_table"
    FEDERATED = "federated"
    SYNTHETIC = "synthetic"
    # Legacy values kept for backward compat
    MANAGED_UC = "managed_uc"
    EXTERNAL_UC = "external_uc"
    UPLOAD = "upload"


class Environment(str, Enum):
    """Deployment environment."""
    DEV = "dev"
    QA = "qa"
    PROD = "prod"


class ApprovalStatus(str, Enum):
    """Status of promotion approval."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class AIModel(str, Enum):
    """Available AI models for conversion."""
    # Cheap/Fast - good for simple tasks
    GPT_NANO = "databricks-gpt-5-nano"
    GPT_OSS_20B = "databricks-gpt-oss-20b"
    GPT_OSS = "databricks-gpt-oss-120b"
    HAIKU = "databricks-claude-haiku-4-5"

    # Complex reasoning - expensive but thorough
    OPUS = "databricks-claude-opus-4-5"


class ReferenceFile(BaseModel):
    """A reference file uploaded for conversion context."""
    file_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_name: str
    file_path: str  # UC Volume path
    file_type: str  # md, sql, txt, etc
    description: Optional[str] = None
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)


class QualityCheck(BaseModel):
    """A quality check result."""
    check_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    check_name: str
    category: str  # syntax, semantics, best_practices, performance
    passed: bool
    severity: str  # info, warning, error
    message: str
    details: Optional[str] = None
    line_number: Optional[int] = None
    suggestion: Optional[str] = None


class QualityReport(BaseModel):
    """Quality assessment report for a conversion."""
    overall_score: float  # 0-100
    checks: List[QualityCheck] = Field(default_factory=list)
    summary: str
    recommendations: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class ConversionJobCreate(BaseModel):
    """Request to create a new conversion job."""
    job_name: str = Field(..., description="Human-readable job name")
    source_type: SourceType = Field(..., description="Type of source file")
    source_path: str = Field(..., description="UC Volume path to source file")
    output_format: OutputFormat = Field(default=OutputFormat.PYSPARK, description="Target output format")
    source_location: SourceLocation = Field(default=SourceLocation.UPLOAD, description="Where the source file came from")
    ai_model: AIModel = Field(default=AIModel.HAIKU, description="AI model to use for conversion")
    conversion_instructions: Optional[str] = Field(default=None, description="Custom instructions for conversion")
    reference_file_ids: Optional[List[str]] = Field(default=None, description="IDs of reference files to include")
    skill_ids: Optional[List[str]] = Field(default=None, description="IDs of skills to apply")
    template_ids: Optional[List[str]] = Field(default=None, description="IDs of templates to use as examples")
    validation_source: Optional[ValidationSource] = Field(default=None, description="Source of validation data")
    validation_table: Optional[str] = Field(default=None, description="UC table path for validation")
    metadata: Optional[Dict[str, str]] = Field(default=None, description="Additional metadata")


class ConversionJobUpdate(BaseModel):
    """Request to update a conversion job."""
    status: Optional[ConversionStatus] = None
    quality_score: Optional[float] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None


class ConversionJob(BaseModel):
    """A conversion job."""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_name: str
    source_type: SourceType
    source_path: str
    output_path: Optional[str] = None
    output_format: OutputFormat = OutputFormat.PYSPARK
    source_location: SourceLocation = SourceLocation.UPLOAD
    status: ConversionStatus = ConversionStatus.PENDING
    quality_score: Optional[float] = None
    ai_model: AIModel = AIModel.HAIKU
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)

    # Conversion context
    conversion_instructions: Optional[str] = None
    reference_files: List[ReferenceFile] = Field(default_factory=list)
    validation_source: Optional[ValidationSource] = None
    validation_table: Optional[str] = None

    # Quality assessment
    quality_report: Optional[QualityReport] = None

    # Runtime fields (not persisted)
    databricks_run_id: Optional[str] = None
    databricks_run_url: Optional[str] = None

    class Config:
        use_enum_values = True


class ConversionFile(BaseModel):
    """A file within a conversion job."""
    file_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str
    source_file: str
    target_file: Optional[str] = None
    file_type: str  # notebook, workflow, sql, dlt
    status: ConversionStatus = ConversionStatus.PENDING
    quality_score: Optional[float] = None
    validation_details: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class PromotionRequest(BaseModel):
    """Request to promote a conversion to next environment."""
    job_id: str
    to_environment: Environment
    notes: Optional[str] = None


class PromotionHistory(BaseModel):
    """Record of a promotion event."""
    promotion_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str
    from_environment: Environment
    to_environment: Environment
    promoted_by: str
    promoted_at: datetime = Field(default_factory=datetime.utcnow)
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    approver: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    notes: Optional[str] = None
    deployed_job_id: Optional[str] = None
    deployed_job_url: Optional[str] = None

    class Config:
        use_enum_values = True


class StatusEvent(BaseModel):
    """Real-time status event for SSE streaming."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str
    event_type: str  # status_change, progress, log, error
    event_data: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.utcnow)
