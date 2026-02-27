"""
Service for running validation and calculating quality scores.

Uses CodeQualityAnalyzer for real multi-dimensional analysis of generated code
instead of hardcoded fake checks.
"""

import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from databricks.sdk import WorkspaceClient

from ..config import Settings
from ..models.validation import (
    ValidationResult,
    QualityScore,
    ScoreDimension,
    ValidationReport,
    ValidationCategory,
    ValidationSeverity,
)
from .code_quality_analyzer import CodeQualityAnalyzer, QualityAnalysis
from .volume_manager import VolumeManager

logger = logging.getLogger(__name__)


# Map analyzer dimension names to ValidationCategory
DIMENSION_TO_CATEGORY = {
    "code_quality": ValidationCategory.CODE_QUALITY,
    "standards": ValidationCategory.STANDARDS,
    "performance": ValidationCategory.PERFORMANCE,
    "parameterization": ValidationCategory.PARAMETERIZATION,
    "verbosity": ValidationCategory.VERBOSITY,
}


class ValidationService:
    """Service for validation and quality scoring using real code analysis."""

    def __init__(self, client: WorkspaceClient, settings: Settings):
        self.client = client
        self.settings = settings
        self.analyzer = CodeQualityAnalyzer()
        self.volume_manager = VolumeManager(client, settings)

    async def run_validation(
        self,
        job_id: str,
        compare_with_source: bool = True,
        source_table: Optional[str] = None,
        output_table: Optional[str] = None,
    ) -> ValidationReport:
        """Run real validation on a conversion job's output.

        Reads the generated code from UC Volumes and runs multi-dimensional
        analysis using CodeQualityAnalyzer.
        """
        logger.info(f"Running validation for job {job_id}")

        # Read the generated code from output path
        code = await self._read_generated_code(job_id)

        if not code:
            logger.warning(f"No generated code found for job {job_id}")
            return self._empty_report(job_id)

        # Run multi-dimensional analysis
        analysis = self.analyzer.analyze(code)

        # Convert analysis results to validation models
        results = self._analysis_to_results(job_id, analysis)
        score = self._analysis_to_score(job_id, analysis)

        # Get job name
        from ..routers.conversions import _jobs_store
        job = _jobs_store.get(job_id)
        job_name = job.job_name if job else "Unknown"

        report = ValidationReport(
            job_id=job_id,
            job_name=job_name,
            score=score,
            results=results,
            output_metrics={
                "code_length": len(code),
                "line_count": len(code.split("\n")),
            },
        )

        logger.info(
            f"Validation complete for {job_id}: "
            f"score={score.overall_score:.2%}, "
            f"grade={score.grade}, "
            f"checks={score.validation_count} "
            f"(passed={score.passed_count}, failed={score.failed_count})"
        )

        return report

    async def _read_generated_code(self, job_id: str) -> str:
        """Read all generated notebook code for a job from UC Volumes."""
        output_base = f"{self.settings.outputs_path}/{job_id}/notebooks"
        all_code = []

        try:
            files = await self.volume_manager.list_files(output_base)
            for file_info in files:
                name = file_info.name if hasattr(file_info, 'name') else file_info.get("name", "")
                path = file_info.path if hasattr(file_info, 'path') else file_info.get("path", "")
                if name.endswith(".py"):
                    content_bytes = await self.volume_manager.read_file(path)
                    all_code.append(content_bytes.decode("utf-8"))
        except Exception as e:
            logger.warning(f"Could not list files at {output_base}: {e}")
            # Try reading the quality report to get code from there
            try:
                report_path = f"{self.settings.outputs_path}/{job_id}/quality_report.json"
                report_bytes = await self.volume_manager.read_file(report_path)
                # The quality_report.json doesn't have the code, but confirms conversion happened
                logger.info(f"Found quality report for {job_id}, attempting notebook read")
            except Exception:
                pass

        return "\n\n".join(all_code) if all_code else ""

    def _analysis_to_results(
        self, job_id: str, analysis: QualityAnalysis
    ) -> List[ValidationResult]:
        """Convert CodeQualityAnalyzer results to ValidationResult models."""
        results = []
        for dimension in analysis.dimensions:
            category = DIMENSION_TO_CATEGORY.get(
                dimension.name, ValidationCategory.CODE_QUALITY
            )
            for check in dimension.checks:
                results.append(ValidationResult(
                    job_id=job_id,
                    check_name=check.name,
                    passed=check.passed,
                    expected=check.expected or None,
                    actual=check.actual or None,
                    message=check.message,
                    severity=ValidationSeverity(check.severity),
                    category=category,
                ))
        return results

    def _analysis_to_score(
        self, job_id: str, analysis: QualityAnalysis
    ) -> QualityScore:
        """Convert CodeQualityAnalyzer results to QualityScore model."""
        # Build category scores (backward-compatible format)
        category_scores = {}
        for dim in analysis.dimensions:
            category_scores[dim.name] = dim.score / 100.0

        # Build ScoreDimension list
        dimensions = []
        for dim in analysis.dimensions:
            passed = sum(1 for c in dim.checks if c.passed)
            failed = len(dim.checks) - passed
            dimensions.append(ScoreDimension(
                name=dim.name,
                score=dim.score,
                weight=dim.weight,
                description=dim.description,
                checks=[
                    {
                        "name": c.name,
                        "passed": c.passed,
                        "message": c.message,
                        "severity": c.severity,
                        "expected": c.expected,
                        "actual": c.actual,
                    }
                    for c in dim.checks
                ],
                passed_count=passed,
                failed_count=failed,
            ))

        return QualityScore(
            job_id=job_id,
            overall_score=analysis.overall_score / 100.0,
            categories=category_scores,
            dimensions=dimensions,
            validation_count=analysis.total_checks,
            passed_count=analysis.passed_checks,
            failed_count=analysis.failed_checks,
        )

    def _empty_report(self, job_id: str) -> ValidationReport:
        """Return a report indicating no code was available to analyze."""
        from ..routers.conversions import _jobs_store
        job = _jobs_store.get(job_id)
        job_name = job.job_name if job else "Unknown"

        return ValidationReport(
            job_id=job_id,
            job_name=job_name,
            score=QualityScore(
                job_id=job_id,
                overall_score=0.0,
                categories={},
                dimensions=[],
                validation_count=0,
                passed_count=0,
                failed_count=0,
                metrics={"error": "No generated code found to analyze"},
            ),
            results=[
                ValidationResult(
                    job_id=job_id,
                    check_name="code_available",
                    passed=False,
                    message="No generated code found in output path - run conversion first",
                    category=ValidationCategory.CODE_QUALITY,
                    severity=ValidationSeverity.ERROR,
                )
            ],
        )

    async def compare_with_source(
        self,
        job_id: str,
        source_table: str,
        output_table: str,
    ) -> Dict[str, Any]:
        """Compare output with source data.

        Reads real comparison results from the validation pipeline if available.
        Otherwise returns a status indicating the pipeline hasn't been run.
        """
        # Try to read real comparison results from UC Volume
        comparison_path = f"{self.settings.validation_path}/{job_id}/comparison_report.json"
        try:
            content_bytes = await self.volume_manager.read_file(comparison_path)
            return json.loads(content_bytes.decode("utf-8"))
        except Exception:
            logger.info(f"No comparison results found at {comparison_path}")

        return {
            "status": "not_run",
            "message": "Data comparison pipeline has not been run. Use 'Run Full Validation Pipeline' to generate real comparison data.",
            "source_row_count": None,
            "output_row_count": None,
            "match_percentage": None,
            "differences": [],
        }
