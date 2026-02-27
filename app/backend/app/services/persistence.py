"""
Delta table persistence service for conversion jobs.

Replaces in-memory storage with Unity Catalog Delta tables.
Uses the SP client for SQL execution (OBO token lacks 'sql' scope).
"""

import logging
import json
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

import httpx
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

from ..config import Settings
from ..auth import UserContext
from ..models.conversion import (
    ConversionJob,
    ConversionStatus,
    QualityReport,
    QualityCheck,
)

logger = logging.getLogger(__name__)


class PersistenceService:
    """Delta table persistence for conversion jobs."""

    def __init__(self, settings: Settings, user: UserContext):
        self.settings = settings
        self.user = user
        self.catalog = settings.catalog
        self.schema = settings.schema_name
        self.warehouse_id = settings.databricks_warehouse_id

        # SP client for SQL execution (OBO token lacks 'sql' scope)
        host = os.environ.get("DATABRICKS_HOST", "")
        if not host.startswith("https://"):
            host = f"https://{host}"
        self.host = host
        self.sp_client = WorkspaceClient(host=host)

    async def save_job(self, job: ConversionJob) -> None:
        """Insert or update job in Delta table."""
        # Prepare quality report JSON
        quality_report_json = None
        if job.quality_report:
            quality_report_json = job.quality_report.model_dump_json()

        # Prepare metadata as JSON string for MAP column
        metadata_json = json.dumps(job.metadata) if job.metadata else "{}"

        # Extract refinement_of from metadata if present
        refinement_of = job.metadata.get("original_job_id") if job.metadata else None

        # Safely extract enum values (Python 3.11+ changes str(Enum) behavior)
        status_val = self._enum_val(job.status)
        source_type_val = self._enum_val(job.source_type)
        ai_model_val = self._enum_val(job.ai_model) if job.ai_model else 'databricks-claude-haiku-4-5'

        query = f"""
        MERGE INTO {self.catalog}.{self.schema}.conversion_jobs AS target
        USING (SELECT
            '{job.job_id}' AS job_id,
            '{self._escape(job.job_name)}' AS job_name,
            '{source_type_val}' AS source_type,
            '{self._escape(job.source_path)}' AS source_path,
            {self._sql_str(job.output_path)} AS output_path,
            '{status_val}' AS status,
            {job.quality_score if job.quality_score else 'NULL'} AS quality_score,
            '{ai_model_val}' AS ai_model,
            {self._sql_str(job.conversion_instructions)} AS conversion_instructions,
            {self._sql_str(refinement_of)} AS refinement_of,
            {self._sql_str(job.created_by)} AS created_by,
            '{job.created_at.isoformat()}' AS created_at,
            current_timestamp() AS updated_at,
            {self._sql_str(job.completed_at.isoformat() if job.completed_at else None)} AS completed_at,
            {self._sql_str(job.error_message)} AS error_message,
            {self._sql_str(job.databricks_run_id)} AS databricks_run_id,
            from_json('{metadata_json}', 'MAP<STRING, STRING>') AS metadata,
            {self._sql_str(quality_report_json)} AS quality_report_json
        ) AS source
        ON target.job_id = source.job_id
        WHEN MATCHED THEN UPDATE SET
            job_name = source.job_name,
            source_type = source.source_type,
            source_path = source.source_path,
            output_path = source.output_path,
            status = source.status,
            quality_score = source.quality_score,
            ai_model = source.ai_model,
            conversion_instructions = source.conversion_instructions,
            refinement_of = source.refinement_of,
            updated_at = source.updated_at,
            completed_at = source.completed_at,
            error_message = source.error_message,
            databricks_run_id = source.databricks_run_id,
            metadata = source.metadata,
            quality_report_json = source.quality_report_json
        WHEN NOT MATCHED THEN INSERT (
            job_id, job_name, source_type, source_path, output_path, status,
            quality_score, ai_model, conversion_instructions, refinement_of,
            created_by, created_at, updated_at, completed_at, error_message,
            databricks_run_id, metadata, quality_report_json
        ) VALUES (
            source.job_id, source.job_name, source.source_type, source.source_path,
            source.output_path, source.status, source.quality_score, source.ai_model,
            source.conversion_instructions, source.refinement_of, source.created_by,
            source.created_at, source.updated_at, source.completed_at, source.error_message,
            source.databricks_run_id, source.metadata, source.quality_report_json
        )
        """

        try:
            await self._execute_sql(query)
            logger.info(f"Saved job {job.job_id} to Delta table")
        except Exception as e:
            logger.error(f"Failed to save job {job.job_id}: {e}")
            raise

    async def get_job(self, job_id: str) -> Optional[ConversionJob]:
        """Get job by ID."""
        query = f"""
        SELECT *
        FROM {self.catalog}.{self.schema}.conversion_jobs
        WHERE job_id = '{job_id}'
        """

        results = await self._execute_sql(query)
        if not results:
            return None

        return self._row_to_job(results[0])

    async def list_jobs(
        self,
        status: Optional[ConversionStatus] = None,
        source_type: Optional[str] = None,
        created_by: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ConversionJob]:
        """List jobs with optional filters."""
        where_clauses = []
        if status:
            where_clauses.append(f"status = '{status}'")
        if source_type:
            where_clauses.append(f"source_type = '{source_type}'")
        if created_by:
            where_clauses.append(f"created_by = '{self._escape(created_by)}'")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        query = f"""
        SELECT *
        FROM {self.catalog}.{self.schema}.conversion_jobs
        {where_sql}
        ORDER BY created_at DESC
        LIMIT {limit} OFFSET {offset}
        """

        results = await self._execute_sql(query)
        return [self._row_to_job(row) for row in results]

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job from the Delta table."""
        query = f"""
        DELETE FROM {self.catalog}.{self.schema}.conversion_jobs
        WHERE job_id = '{job_id}'
        """

        try:
            await self._execute_sql(query)
            logger.info(f"Deleted job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete job {job_id}: {e}")
            return False

    async def get_job_lineage(self, job_id: str) -> Dict[str, Any]:
        """Get full lineage chain for a job.

        Returns the original job and all its refinements in order.
        """
        # First, find the root of the lineage (original job)
        root_query = f"""
        WITH RECURSIVE ancestors AS (
            SELECT job_id, refinement_of, 0 AS depth
            FROM {self.catalog}.{self.schema}.conversion_jobs
            WHERE job_id = '{job_id}'

            UNION ALL

            SELECT c.job_id, c.refinement_of, a.depth + 1
            FROM {self.catalog}.{self.schema}.conversion_jobs c
            JOIN ancestors a ON c.job_id = a.refinement_of
            WHERE c.refinement_of IS NOT NULL
        )
        SELECT job_id FROM ancestors
        WHERE refinement_of IS NULL
        LIMIT 1
        """

        root_results = await self._execute_sql(root_query)
        if not root_results:
            # Job has no lineage, it's the root
            root_job_id = job_id
        else:
            root_job_id = root_results[0].get("job_id", job_id)

        # Now get all descendants of the root
        descendants_query = f"""
        WITH RECURSIVE lineage AS (
            SELECT *
            FROM {self.catalog}.{self.schema}.conversion_jobs
            WHERE job_id = '{root_job_id}'

            UNION ALL

            SELECT c.*
            FROM {self.catalog}.{self.schema}.conversion_jobs c
            JOIN lineage l ON c.refinement_of = l.job_id
        )
        SELECT * FROM lineage ORDER BY created_at
        """

        results = await self._execute_sql(descendants_query)
        jobs = [self._row_to_job(row) for row in results]

        if not jobs:
            return {"original": None, "refinements": []}

        return {
            "original": jobs[0],
            "refinements": jobs[1:] if len(jobs) > 1 else [],
        }

    async def get_refinements(self, job_id: str) -> List[ConversionJob]:
        """Get all direct refinements of a job."""
        query = f"""
        SELECT *
        FROM {self.catalog}.{self.schema}.conversion_jobs
        WHERE refinement_of = '{job_id}'
        ORDER BY created_at
        """

        results = await self._execute_sql(query)
        return [self._row_to_job(row) for row in results]

    def _row_to_job(self, row: Dict[str, Any]) -> ConversionJob:
        """Convert a SQL row to ConversionJob."""
        # Parse quality report JSON if present
        quality_report = None
        quality_report_json = row.get("quality_report_json")
        if quality_report_json:
            try:
                qr_data = json.loads(quality_report_json)
                quality_report = QualityReport(**qr_data)
            except Exception as e:
                logger.warning(f"Failed to parse quality_report_json: {e}")

        # Parse metadata MAP
        metadata = row.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}

        return ConversionJob(
            job_id=row["job_id"],
            job_name=row["job_name"],
            source_type=row["source_type"],
            source_path=row["source_path"],
            output_path=row.get("output_path"),
            status=row["status"],
            quality_score=row.get("quality_score"),
            ai_model=row.get("ai_model", "databricks-claude-haiku-4-5"),
            created_by=row.get("created_by"),
            created_at=self._parse_datetime(row.get("created_at")),
            updated_at=self._parse_datetime(row.get("updated_at")),
            completed_at=self._parse_datetime(row.get("completed_at")),
            error_message=row.get("error_message"),
            metadata=metadata,
            conversion_instructions=row.get("conversion_instructions"),
            quality_report=quality_report,
            databricks_run_id=row.get("databricks_run_id"),
        )

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """Parse datetime from various formats."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except:
                return None
        return None

    def _enum_val(self, value) -> str:
        """Extract the string value from an enum or return str as-is."""
        if hasattr(value, 'value'):
            return str(value.value)
        return str(value)

    def _escape(self, value: str) -> str:
        """Escape single quotes for SQL."""
        if value is None:
            return ""
        return value.replace("'", "''")

    def _sql_str(self, value: Optional[str]) -> str:
        """Format value as SQL string or NULL."""
        if value is None:
            return "NULL"
        return f"'{self._escape(value)}'"

    def _get_warehouse_id(self) -> str:
        """Get warehouse ID from settings or environment."""
        wh = self.warehouse_id
        if not wh:
            # app.yaml valueFrom: sql-warehouse sets DATABRICKS_WAREHOUSE_ID
            # but also check alternative env var names
            wh = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")
        if not wh:
            # The sql-warehouse resource binding may set this
            wh = os.environ.get("SQL_WAREHOUSE_ID", "")
        if not wh:
            logger.error("No warehouse_id configured. Set DATABRICKS_WAREHOUSE_ID env var.")
            raise RuntimeError("No SQL Warehouse ID configured. Set DATABRICKS_WAREHOUSE_ID environment variable.")
        return wh

    async def _execute_sql(self, query: str) -> List[Dict[str, Any]]:
        """Execute SQL via Statement Execution API using SP credentials.

        Uses the SP client because OBO tokens in Databricks Apps lack 'sql' scope.
        """
        import asyncio

        warehouse_id = self._get_warehouse_id()

        def run_query():
            statement = self.sp_client.statement_execution.execute_statement(
                warehouse_id=warehouse_id,
                statement=query,
                wait_timeout="30s",
            )

            # Wait for completion
            while statement.status.state in (
                StatementState.PENDING,
                StatementState.RUNNING,
            ):
                import time
                time.sleep(0.5)
                statement = self.sp_client.statement_execution.get_statement(
                    statement.statement_id
                )

            if statement.status.state == StatementState.FAILED:
                error = statement.status.error
                raise Exception(f"SQL failed: {error.message if error else 'Unknown error'}")

            # Extract results
            if not statement.result or not statement.result.data_array:
                return []

            # Map columns to row dicts
            columns = [c.name for c in statement.manifest.schema.columns]
            rows = []
            for data_row in statement.result.data_array:
                row_dict = dict(zip(columns, data_row))
                rows.append(row_dict)

            return rows

        return await asyncio.to_thread(run_query)


# Factory function for dependency injection
def get_persistence_service(
    settings: Settings, user: UserContext
) -> PersistenceService:
    """Get PersistenceService instance."""
    return PersistenceService(settings, user)
