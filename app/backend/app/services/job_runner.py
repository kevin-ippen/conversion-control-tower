"""
Service for running conversions as Databricks Jobs.

Uses a pre-configured job bound to the app as a resource.
The app triggers runs with parameters via run_now().
"""

import logging
import os
from typing import Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import RunParameters

from ..config import Settings

logger = logging.getLogger(__name__)


class JobRunnerError(Exception):
    """Custom exception for job runner errors with detailed context."""
    pass


class JobRunner:
    """Runs conversions using a pre-configured Databricks Job.

    The job is bound to the app as a resource, giving the app's
    service principal permission to trigger runs.
    """

    def __init__(self, client: WorkspaceClient, settings: Settings):
        self.user_client = client  # User's OBO client (for audit trail)
        self.settings = settings

        # Create a service principal client for job operations
        host = os.environ.get("DATABRICKS_HOST", "")
        if not host.startswith("https://"):
            host = f"https://{host}"

        # WorkspaceClient() will use the SP credentials from env vars
        self.client = WorkspaceClient(host=host)
        logger.info(f"JobRunner init - SDK auth_type: {self.client.config.auth_type}")

        # Get the pre-configured job IDs from environment
        self.conversion_job_id = os.environ.get("CONVERSION_JOB_ID", "")
        self.original_simulator_job_id = os.environ.get("ORIGINAL_SIMULATOR_JOB_ID", "")
        self.converted_runner_job_id = os.environ.get("CONVERTED_RUNNER_JOB_ID", "")
        self.data_comparator_job_id = os.environ.get("DATA_COMPARATOR_JOB_ID", "")

        if not self.conversion_job_id:
            logger.warning("CONVERSION_JOB_ID not set - job runs will fail")
        else:
            logger.info(f"JobRunner using conversion job ID: {self.conversion_job_id}")

        if all([self.original_simulator_job_id, self.converted_runner_job_id, self.data_comparator_job_id]):
            logger.info(
                f"Validation pipeline jobs: simulator={self.original_simulator_job_id}, "
                f"runner={self.converted_runner_job_id}, comparator={self.data_comparator_job_id}"
            )
        else:
            logger.warning("Validation pipeline job IDs not fully configured")

    async def submit_conversion_job(
        self,
        job_id: str,
        source_path: str,
        ai_model: str = "databricks-claude-haiku-4-5",
        source_type: str = "ssis",
        output_format: str = "pyspark",
    ) -> str:
        """Trigger a conversion run using the pre-configured job.

        Uses run_now() to trigger the bound job with notebook parameters.

        Args:
            job_id: Unique conversion job identifier (for tracking)
            source_path: UC Volume path to source file
            ai_model: FMAPI model to use for conversion
            source_type: Type of source (ssis, sql_script, stored_proc, informatica_pc)
            output_format: Target output format (pyspark, dlt_sdp, dbt)

        Returns:
            Databricks run_id for monitoring

        Raises:
            JobRunnerError: If job trigger fails
        """
        import asyncio

        if not self.conversion_job_id:
            raise JobRunnerError(
                "CONVERSION_JOB_ID environment variable not set. "
                "Ensure the job is bound as a resource in app.yaml."
            )

        logger.info(f"Triggering conversion job {job_id} for {source_path}")
        logger.info(f"Using model: {ai_model}, source_type: {source_type}, output_format: {output_format}")
        logger.info(f"Databricks job ID: {self.conversion_job_id}")

        output_path = f"{self.settings.outputs_path}/{job_id}"

        # Job-level parameters (must match parameters defined in databricks.yml)
        job_params = {
            "job_id": job_id,
            "source_path": source_path,
            "output_path": output_path,
            "ai_model": ai_model,
            "source_type": source_type,
            "output_format": output_format,
        }

        try:
            # Use run_now() to trigger the pre-configured job
            run = await asyncio.to_thread(
                self.client.jobs.run_now,
                job_id=int(self.conversion_job_id),
                job_parameters=job_params,
            )

            run_id = str(run.run_id)
            logger.info(f"Successfully triggered conversion run {run_id} for job {job_id}")
            return run_id

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to trigger conversion job {job_id}: {error_msg}")

            # Provide detailed error information
            if "403" in error_msg:
                raise JobRunnerError(
                    f"Permission denied (403). Ensure the job is bound as a resource in app.yaml "
                    f"and the service principal has CAN_MANAGE_RUN permission. Error: {error_msg}"
                )
            elif "404" in error_msg:
                raise JobRunnerError(
                    f"Job not found (404). Job ID {self.conversion_job_id} may not exist. "
                    f"Error: {error_msg}"
                )
            elif "Invalid scope" in error_msg:
                raise JobRunnerError(
                    f"Invalid OAuth scope. Ensure the job is bound as a resource in app.yaml. "
                    f"Error: {error_msg}"
                )
            else:
                raise JobRunnerError(f"Job trigger failed: {error_msg}")

    async def get_run_status(self, run_id: str) -> dict:
        """Get status of a job run.

        Args:
            run_id: Databricks run ID

        Returns:
            Dict with status, progress, and other metadata
        """
        import asyncio

        try:
            run = await asyncio.to_thread(
                self.client.jobs.get_run,
                run_id=int(run_id)
            )

            # Calculate progress based on state
            state = run.state.life_cycle_state.value if run.state else "UNKNOWN"
            progress = 0
            if state == "PENDING":
                progress = 10
            elif state == "RUNNING":
                progress = 50
            elif state == "TERMINATING":
                progress = 90
            elif state == "TERMINATED":
                progress = 100

            return {
                "run_id": run_id,
                "state": state,
                "state_message": run.state.state_message if run.state else "",
                "result_state": run.state.result_state.value if run.state and run.state.result_state else None,
                "progress": progress,
                "start_time": run.start_time,
                "end_time": run.end_time,
                "run_page_url": run.run_page_url,
            }

        except Exception as e:
            logger.error(f"Failed to get run status for {run_id}: {e}")
            return {
                "run_id": run_id,
                "state": "ERROR",
                "state_message": str(e),
                "progress": 0,
            }

    async def get_run_output(self, run_id: str) -> Optional[dict]:
        """Get the output of a completed job run.

        Args:
            run_id: Databricks run ID

        Returns:
            Parsed notebook output or None
        """
        import asyncio

        try:
            output = await asyncio.to_thread(
                self.client.jobs.get_run_output,
                run_id=int(run_id)
            )
            if output.notebook_output and output.notebook_output.result:
                import json
                return json.loads(output.notebook_output.result)
            return None
        except Exception as e:
            logger.error(f"Failed to get run output for {run_id}: {e}")
            return None

    async def cancel_run(self, run_id: str) -> bool:
        """Cancel a running job.

        Args:
            run_id: Databricks run ID

        Returns:
            True if cancelled successfully
        """
        import asyncio

        try:
            await asyncio.to_thread(
                self.client.jobs.cancel_run,
                run_id=int(run_id)
            )
            logger.info(f"Cancelled run {run_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel run {run_id}: {e}")
            return False

    # =========================================================================
    # Validation Pipeline
    # =========================================================================

    async def _trigger_job(self, dabs_job_id: str, job_params: dict) -> str:
        """Trigger a DABs-deployed job with job-level parameters."""
        import asyncio

        run = await asyncio.to_thread(
            self.client.jobs.run_now,
            job_id=int(dabs_job_id),
            job_parameters=job_params,
        )
        return str(run.run_id)

    async def _wait_for_run(self, run_id: str, poll_interval: int = 10) -> dict:
        """Wait for a job run to complete and return its final status."""
        import asyncio

        while True:
            status = await self.get_run_status(run_id)
            state = status.get("state", "UNKNOWN")

            if state in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
                return status

            await asyncio.sleep(poll_interval)

    async def submit_validation_pipeline(
        self,
        conversion_job_id: str,
        source_path: str,
        output_path: str,
        validation_table: str | None = None,
    ) -> dict:
        """Run the full 3-step validation pipeline.

        Steps:
        1. Original simulator - generates expected output from SSIS logic
           (or reads from a real UC table if validation_table is provided)
        2. Converted runner - executes converted Databricks code
        3. Data comparator - compares expected vs actual output

        Args:
            conversion_job_id: The conversion job ID
            source_path: Path to the uploaded source file
            output_path: Path to the converted output
            validation_table: Optional UC table (catalog.schema.table) for real source data

        Returns:
            Dict with run_ids and status for each step
        """
        if not all([self.original_simulator_job_id, self.converted_runner_job_id, self.data_comparator_job_id]):
            raise JobRunnerError(
                "Validation pipeline job IDs not configured. "
                "Set ORIGINAL_SIMULATOR_JOB_ID, CONVERTED_RUNNER_JOB_ID, and DATA_COMPARATOR_JOB_ID in app.yaml."
            )

        validation_base = f"{self.settings.validation_path}/{conversion_job_id}"
        expected_path = f"{validation_base}/expected_output"
        actual_path = f"{validation_base}/actual_output"
        source_data_path = f"{validation_base}/source_data"
        compare_output_path = validation_base

        result = {
            "conversion_job_id": conversion_job_id,
            "steps": {},
            "status": "running",
        }

        try:
            # Step 1: Original Simulator
            # Extracts schemas from source file, generates synthetic data,
            # and simulates expected output
            logger.info(f"Pipeline step 1/3: Original Simulator for {conversion_job_id}")
            sim_params = {
                "conversion_job_id": conversion_job_id,
                "source_file_path": source_path,
                "output_path": expected_path,
                "source_data_output_path": source_data_path,
            }
            if validation_table:
                sim_params["validation_table"] = validation_table
            sim_run_id = await self._trigger_job(
                self.original_simulator_job_id,
                sim_params,
            )
            result["steps"]["original_simulator"] = {"run_id": sim_run_id, "status": "running"}

            sim_status = await self._wait_for_run(sim_run_id)
            sim_result = sim_status.get("result_state", "UNKNOWN")
            result["steps"]["original_simulator"]["status"] = sim_result
            result["steps"]["original_simulator"]["run_page_url"] = sim_status.get("run_page_url")

            if sim_result != "SUCCESS":
                result["status"] = "failed"
                result["error"] = f"Original simulator failed: {sim_status.get('state_message', '')}"
                logger.error(f"Pipeline step 1 failed: {sim_result}")
                return result

            # Step 2: Converted Runner
            # Reads synthetic data from Step 1, executes converted notebooks
            logger.info(f"Pipeline step 2/3: Converted Runner for {conversion_job_id}")
            runner_run_id = await self._trigger_job(
                self.converted_runner_job_id,
                {
                    "conversion_job_id": conversion_job_id,
                    "source_data_path": source_data_path,
                    "converted_notebook_path": f"{output_path}/notebooks",
                    "output_path": actual_path,
                },
            )
            result["steps"]["converted_runner"] = {"run_id": runner_run_id, "status": "running"}

            runner_status = await self._wait_for_run(runner_run_id)
            runner_result = runner_status.get("result_state", "UNKNOWN")
            result["steps"]["converted_runner"]["status"] = runner_result
            result["steps"]["converted_runner"]["run_page_url"] = runner_status.get("run_page_url")

            if runner_result != "SUCCESS":
                result["status"] = "failed"
                result["error"] = f"Converted runner failed: {runner_status.get('state_message', '')}"
                logger.error(f"Pipeline step 2 failed: {runner_result}")
                return result

            # Step 3: Data Comparator
            logger.info(f"Pipeline step 3/3: Data Comparator for {conversion_job_id}")
            comp_run_id = await self._trigger_job(
                self.data_comparator_job_id,
                {
                    "conversion_job_id": conversion_job_id,
                    "expected_path": expected_path,
                    "actual_path": actual_path,
                    "output_path": compare_output_path,
                },
            )
            result["steps"]["data_comparator"] = {"run_id": comp_run_id, "status": "running"}

            comp_status = await self._wait_for_run(comp_run_id)
            comp_result = comp_status.get("result_state", "UNKNOWN")
            result["steps"]["data_comparator"]["status"] = comp_result
            result["steps"]["data_comparator"]["run_page_url"] = comp_status.get("run_page_url")

            if comp_result == "SUCCESS":
                result["status"] = "completed"
                logger.info(f"Validation pipeline completed for {conversion_job_id}")
            else:
                result["status"] = "failed"
                result["error"] = f"Data comparator failed: {comp_status.get('state_message', '')}"

            return result

        except Exception as e:
            logger.error(f"Validation pipeline failed for {conversion_job_id}: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
            return result
