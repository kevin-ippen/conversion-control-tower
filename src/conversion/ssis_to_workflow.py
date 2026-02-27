"""
SSIS to Databricks Workflow Converter

Converts SSIS control flow to Databricks Jobs API workflow definitions.
"""

import json
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path

from src.ssis.dtsx_parser import (
    SSISPackage,
    SSISTask,
    SSISConnection,
    SSISPrecedenceConstraint,
    SSISEventHandler
)


@dataclass
class WorkflowTask:
    """Represents a task in a Databricks workflow."""
    task_key: str
    task_type: str  # sql_task, notebook_task, run_job_task, etc.
    task_config: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)
    description: str = ""
    timeout_seconds: int = 3600
    max_retries: int = 1
    min_retry_interval_millis: int = 60000


@dataclass
class DatabricksWorkflow:
    """Represents a complete Databricks workflow."""
    name: str
    description: str
    tasks: List[WorkflowTask] = field(default_factory=list)
    job_clusters: List[Dict] = field(default_factory=list)
    parameters: List[Dict] = field(default_factory=list)
    schedule: Optional[Dict] = None
    email_notifications: Optional[Dict] = None
    tags: Dict[str, str] = field(default_factory=dict)


class SSISToWorkflowConverter:
    """
    Converts SSIS packages to Databricks Workflow definitions.
    """

    def __init__(self, mappings_path: Optional[Path] = None):
        """Initialize converter with optional custom mappings."""
        if mappings_path is None:
            mappings_path = Path(__file__).parent.parent.parent / "config" / "ssis_component_mappings.yaml"

        self.mappings = {}
        if mappings_path.exists():
            with open(mappings_path) as f:
                self.mappings = yaml.safe_load(f)

        self.task_mappings = self.mappings.get("control_flow_tasks", {})

    def convert(self, package: SSISPackage) -> DatabricksWorkflow:
        """Convert an SSIS package to a Databricks workflow."""

        # Convert tasks
        workflow_tasks = []
        task_key_map = {}  # Maps SSIS ref_id to workflow task_key

        for task in package.tasks:
            converted = self._convert_task(task, package)
            workflow_tasks.extend(converted)

            # Track mapping for dependency resolution
            if converted:
                task_key_map[task.ref_id] = converted[0].task_key

        # Resolve dependencies from precedence constraints
        self._resolve_dependencies(
            workflow_tasks,
            package.precedence_constraints,
            task_key_map,
            package.tasks
        )

        # Convert parameters from variables
        parameters = self._convert_parameters(package)

        # Convert event handlers to notifications
        notifications = self._convert_event_handlers(package)

        # Generate job clusters
        job_clusters = self._generate_job_clusters(package)

        return DatabricksWorkflow(
            name=f"converted_{self._sanitize_name(package.name)}",
            description=f"Converted from SSIS package: {package.name}\n\n{package.description}",
            tasks=workflow_tasks,
            job_clusters=job_clusters,
            parameters=parameters,
            email_notifications=notifications,
            tags={
                "source": "ssis_conversion",
                "original_package": package.name
            }
        )

    def _convert_task(
        self,
        task: SSISTask,
        package: SSISPackage,
        parent_key: str = ""
    ) -> List[WorkflowTask]:
        """Convert a single SSIS task to Databricks task(s)."""

        mapping = self.task_mappings.get(task.task_type, {})
        target = mapping.get("target", "notebook_task")

        task_key = self._sanitize_name(task.name)
        if parent_key:
            task_key = f"{parent_key}__{task_key}"

        converted_tasks = []

        if task.task_type == "ExecuteSQLTask":
            converted_tasks.append(self._convert_sql_task(task, task_key))

        elif task.task_type == "DataFlowTask":
            converted_tasks.append(self._convert_data_flow_task(task, task_key))

        elif task.task_type == "SendMailTask":
            # Convert to notification - handled at workflow level
            # Add a placeholder task that can trigger notification
            converted_tasks.append(WorkflowTask(
                task_key=task_key,
                task_type="notebook_task",
                task_config={
                    "notebook_path": f"/Repos/converted/{package.name}/utilities/send_notification",
                    "base_parameters": {
                        "to": task.properties.get("to", ""),
                        "subject": task.properties.get("subject", ""),
                        "message": task.properties.get("message", "")
                    }
                },
                description=f"Email notification: {task.properties.get('subject', '')}"
            ))

        elif task.task_type in ("SequenceContainer", "ForLoopContainer", "ForEachLoopContainer"):
            # Convert container's child tasks
            for child in task.child_tasks:
                child_tasks = self._convert_task(child, package, task_key)
                converted_tasks.extend(child_tasks)

            # Resolve internal dependencies
            if task.precedence_constraints:
                child_key_map = {t.ref_id: self._sanitize_name(t.name) for t in task.child_tasks}
                self._resolve_dependencies(
                    converted_tasks,
                    task.precedence_constraints,
                    child_key_map,
                    task.child_tasks
                )

        else:
            # Generic notebook task for unknown types
            converted_tasks.append(WorkflowTask(
                task_key=task_key,
                task_type="notebook_task",
                task_config={
                    "notebook_path": f"/Repos/converted/{package.name}/tasks/{task_key}",
                    "base_parameters": {}
                },
                description=f"Converted from {task.task_type}: {task.name}"
            ))

        return converted_tasks

    def _convert_sql_task(self, task: SSISTask, task_key: str) -> WorkflowTask:
        """Convert Execute SQL Task to Databricks SQL task."""

        sql_statement = task.properties.get("sql_statement", "")

        # Clean up SQL - basic transformations
        sql_statement = self._transform_sql(sql_statement)

        return WorkflowTask(
            task_key=task_key,
            task_type="sql_task",
            task_config={
                "warehouse_id": "{{warehouse_id}}",  # Template variable
                "query": {
                    "query_text": sql_statement
                }
            },
            description=f"SQL Task: {task.name}"
        )

    def _convert_data_flow_task(self, task: SSISTask, task_key: str) -> WorkflowTask:
        """Convert Data Flow Task to notebook task."""

        return WorkflowTask(
            task_key=task_key,
            task_type="notebook_task",
            task_config={
                "notebook_path": f"/Repos/converted/notebooks/{task_key}",
                "base_parameters": {
                    "execution_date": "{{job.start_time.iso_date}}"
                }
            },
            description=f"Data Flow: {task.name}",
            timeout_seconds=7200  # Data flows may take longer
        )

    def _transform_sql(self, sql: str) -> str:
        """Apply basic SQL Server to Spark SQL transformations."""

        # These are simple transformations - complex ones need Claude API
        transformations = [
            ("GETDATE()", "current_timestamp()"),
            ("GETUTCDATE()", "current_timestamp()"),
            ("ISNULL(", "COALESCE("),
            ("NOLOCK", ""),
            ("WITH (NOLOCK)", ""),
            ("dbo.", ""),  # Remove schema prefix - use full UC path
        ]

        result = sql
        for old, new in transformations:
            result = result.replace(old, new)

        return result

    def _resolve_dependencies(
        self,
        tasks: List[WorkflowTask],
        constraints: List[SSISPrecedenceConstraint],
        key_map: Dict[str, str],
        ssis_tasks: List[SSISTask]
    ):
        """Resolve task dependencies from precedence constraints."""

        # Build ref_id to task_key mapping
        ref_to_key = {}
        for t in ssis_tasks:
            ref_to_key[t.ref_id] = self._sanitize_name(t.name)

        # Build task_key lookup for workflow tasks
        task_lookup = {t.task_key: t for t in tasks}

        for constraint in constraints:
            from_key = ref_to_key.get(constraint.from_task, "")
            to_key = ref_to_key.get(constraint.to_task, "")

            if from_key and to_key and to_key in task_lookup:
                # Find the from_key in task_lookup (may have prefix)
                matching_from = None
                for tk in task_lookup:
                    if tk == from_key or tk.endswith(f"__{from_key}"):
                        matching_from = tk
                        break

                matching_to = None
                for tk in task_lookup:
                    if tk == to_key or tk.endswith(f"__{to_key}"):
                        matching_to = tk
                        break

                if matching_from and matching_to:
                    if matching_from not in task_lookup[matching_to].depends_on:
                        task_lookup[matching_to].depends_on.append(matching_from)

    def _convert_parameters(self, package: SSISPackage) -> List[Dict]:
        """Convert SSIS variables to job parameters."""

        parameters = []

        for var in package.variables:
            if var.namespace == "User":
                param = {
                    "name": var.name,
                    "default": str(var.value) if var.value else ""
                }
                parameters.append(param)

        return parameters

    def _convert_event_handlers(self, package: SSISPackage) -> Optional[Dict]:
        """Convert SSIS event handlers to job notifications."""

        notifications = {
            "on_start": [],
            "on_success": [],
            "on_failure": []
        }

        for handler in package.event_handlers:
            if handler.event_name == "OnError":
                # Look for SendMailTask in error handler
                for task in handler.tasks:
                    if task.task_type == "SendMailTask":
                        to_addresses = task.properties.get("to", "").split(";")
                        notifications["on_failure"].extend(to_addresses)

        # Only return if we have notifications
        if any(notifications.values()):
            return {"email_notifications": notifications}

        return None

    def _generate_job_clusters(self, package: SSISPackage) -> List[Dict]:
        """Generate job cluster configurations."""

        return [{
            "job_cluster_key": "main_cluster",
            "new_cluster": {
                "spark_version": "14.3.x-scala2.12",
                "node_type_id": "Standard_DS3_v2",
                "num_workers": 2,
                "spark_conf": {
                    "spark.databricks.delta.optimizeWrite.enabled": "true",
                    "spark.databricks.adaptive.enabled": "true"
                }
            }
        }]

    def _sanitize_name(self, name: str) -> str:
        """Convert name to valid Databricks task key."""

        # Replace invalid characters
        result = name.replace(" ", "_")
        result = result.replace("-", "_")
        result = result.replace(".", "_")
        result = result.replace("(", "")
        result = result.replace(")", "")

        # Ensure starts with letter
        if result and not result[0].isalpha():
            result = f"task_{result}"

        return result.lower()

    def to_json(self, workflow: DatabricksWorkflow) -> str:
        """Serialize workflow to Databricks Jobs API JSON format."""

        job_def = {
            "name": workflow.name,
            "description": workflow.description,
            "tags": workflow.tags,
            "tasks": [],
            "job_clusters": workflow.job_clusters,
            "parameters": workflow.parameters,
            "max_concurrent_runs": 1
        }

        # Add tasks
        for task in workflow.tasks:
            task_def = {
                "task_key": task.task_key,
                "description": task.description,
                "timeout_seconds": task.timeout_seconds,
                "max_retries": task.max_retries,
                "min_retry_interval_millis": task.min_retry_interval_millis
            }

            # Add task-specific config
            if task.task_type == "sql_task":
                task_def["sql_task"] = task.task_config
            elif task.task_type == "notebook_task":
                task_def["notebook_task"] = task.task_config
                task_def["job_cluster_key"] = "main_cluster"
            elif task.task_type == "run_job_task":
                task_def["run_job_task"] = task.task_config
            else:
                task_def["notebook_task"] = task.task_config
                task_def["job_cluster_key"] = "main_cluster"

            # Add dependencies
            if task.depends_on:
                task_def["depends_on"] = [
                    {"task_key": dep} for dep in task.depends_on
                ]

            job_def["tasks"].append(task_def)

        # Add notifications
        if workflow.email_notifications:
            job_def["email_notifications"] = workflow.email_notifications.get(
                "email_notifications", {}
            )

        # Add schedule if present
        if workflow.schedule:
            job_def["schedule"] = workflow.schedule

        return json.dumps(job_def, indent=2)
