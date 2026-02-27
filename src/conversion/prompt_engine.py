"""
Prompt Template Engine

Builds prompts for Claude API using configuration, rules, and examples.
Supports example-based learning and declarative rule enforcement.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from src.config_loader import ConversionConfig, ConversionExample, ExampleMatcher
from src.ssis.dtsx_parser import (
    SSISPackage,
    SSISTask,
    SSISDataFlow,
    SSISDataFlowComponent
)


@dataclass
class ConversionPrompt:
    """A structured prompt for the conversion API."""
    system_prompt: str
    user_prompt: str
    examples: List[Dict[str, str]]
    metadata: Dict[str, Any]


class PromptEngine:
    """
    Builds conversion prompts using configuration and examples.
    """

    def __init__(self, config: ConversionConfig):
        self.config = config
        self.example_matcher = ExampleMatcher(config.examples)

    def build_component_conversion_prompt(
        self,
        component: SSISDataFlowComponent,
        context: Optional[Dict[str, Any]] = None
    ) -> ConversionPrompt:
        """Build a prompt to convert a single data flow component."""

        # Find relevant examples
        relevant_examples = self.example_matcher.find_matches(
            component_type=component.component_type,
            tags=self._get_context_tags(component),
            max_results=2
        )

        # Build system prompt
        system_prompt = self._build_system_prompt(
            conversion_type="component",
            component_type=component.component_type
        )

        # Build user prompt with component details
        user_prompt = self._build_component_user_prompt(component, context)

        # Format examples for few-shot learning
        formatted_examples = self._format_examples(relevant_examples)

        return ConversionPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            examples=formatted_examples,
            metadata={
                "component_type": component.component_type,
                "component_name": component.name,
                "example_count": len(relevant_examples)
            }
        )

    def build_data_flow_conversion_prompt(
        self,
        data_flow: SSISDataFlow,
        context: Optional[Dict[str, Any]] = None
    ) -> ConversionPrompt:
        """Build a prompt to convert an entire data flow."""

        # Collect component types for example matching
        component_types = list(set(c.component_type for c in data_flow.components))

        # Find examples for each component type
        all_examples = []
        for comp_type in component_types[:3]:  # Limit to avoid token overflow
            examples = self.example_matcher.find_matches(
                component_type=comp_type,
                max_results=1
            )
            all_examples.extend(examples)

        system_prompt = self._build_system_prompt(
            conversion_type="data_flow"
        )

        user_prompt = self._build_data_flow_user_prompt(data_flow, context)

        return ConversionPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            examples=self._format_examples(all_examples),
            metadata={
                "data_flow_name": data_flow.name,
                "component_count": len(data_flow.components),
                "component_types": component_types
            }
        )

    def build_workflow_conversion_prompt(
        self,
        package: SSISPackage,
        context: Optional[Dict[str, Any]] = None
    ) -> ConversionPrompt:
        """Build a prompt to convert control flow to Databricks workflow."""

        # Find workflow examples
        workflow_examples = self.example_matcher.find_matches(
            tags=["workflow", "control_flow", "job", "orchestration"],
            max_results=2
        )

        system_prompt = self._build_system_prompt(
            conversion_type="workflow"
        )

        user_prompt = self._build_workflow_user_prompt(package, context)

        return ConversionPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            examples=self._format_examples(workflow_examples),
            metadata={
                "package_name": package.name,
                "task_count": len(package.tasks)
            }
        )

    def build_sql_conversion_prompt(
        self,
        sql_code: str,
        source_type: str = "stored_procedure",
        context: Optional[Dict[str, Any]] = None
    ) -> ConversionPrompt:
        """Build a prompt to convert SQL Server code to Spark SQL."""

        system_prompt = self._build_system_prompt(
            conversion_type="sql",
            source_type=source_type
        )

        user_prompt = f"""Convert the following SQL Server {source_type} to Databricks Spark SQL.

## Source SQL
```sql
{sql_code}
```

## Requirements
- Convert to Spark SQL dialect
- Use Unity Catalog three-part naming: {self.config.default_catalog}.{self.config.default_schema}.<table>
- Apply data type conversions as needed
- Handle SQL Server specific functions
- Add appropriate comments

## Output
Generate the converted Spark SQL code with explanations for any significant changes.
"""

        return ConversionPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            examples=[],
            metadata={
                "source_type": source_type,
                "sql_length": len(sql_code)
            }
        )

    def _build_system_prompt(
        self,
        conversion_type: str,
        component_type: Optional[str] = None,
        source_type: Optional[str] = None
    ) -> str:
        """Build the system prompt with rules and instructions."""

        # Base system prompt
        system_parts = [
            "You are an expert data engineer specializing in migrating SQL Server and SSIS workloads to Databricks.",
            "",
            "## Your Capabilities",
            "- Deep understanding of SSIS package structure and components",
            "- Expert in PySpark, Spark SQL, and Delta Lake",
            "- Knowledge of Databricks Jobs API and workflow orchestration",
            "- Experience with data warehouse migration patterns",
            "",
        ]

        # Add global instructions
        if self.config.global_instructions:
            system_parts.extend([
                "## Global Conversion Rules",
                self.config.global_instructions,
                ""
            ])

        # Add component-specific instructions
        if component_type and component_type in self.config.component_instructions:
            system_parts.extend([
                f"## {component_type} Specific Rules",
                self.config.component_instructions[component_type],
                ""
            ])

        # Add conversion-type specific instructions
        if conversion_type == "component":
            system_parts.extend([
                "## Component Conversion Guidelines",
                "- Generate clean, idiomatic PySpark code",
                "- Use DataFrame API over RDD operations",
                "- Prefer built-in functions over UDFs",
                "- Handle null values explicitly",
                "- Add comments explaining the business logic",
                ""
            ])
        elif conversion_type == "data_flow":
            system_parts.extend([
                "## Data Flow Conversion Guidelines",
                "- Convert to a complete PySpark notebook",
                "- Maintain data lineage through the pipeline",
                "- Use caching for DataFrames used multiple times",
                "- Handle errors with try/except and quarantine tables",
                "- Include logging for row counts and metrics",
                ""
            ])
        elif conversion_type == "workflow":
            system_parts.extend([
                "## Workflow Conversion Guidelines",
                "- Generate Databricks Jobs API JSON format",
                "- Map control flow to task dependencies",
                "- Convert variables to job parameters",
                "- Set appropriate timeouts and retries",
                "- Configure notifications for failures",
                ""
            ])

        # Add business rules if present
        if self.config.business_rules:
            system_parts.extend([
                "## Business Rules",
                self.config.business_rules,
                ""
            ])

        # Add quality requirements
        system_parts.extend([
            "## Quality Requirements",
            "- Generated code must be syntactically correct",
            "- No hardcoded credentials - use Databricks Secrets",
            "- Use Unity Catalog three-part naming",
            "- Follow Python/PySpark best practices",
            f"- Target Databricks Runtime: {self.config.runtime_version}",
            ""
        ])

        return "\n".join(system_parts)

    def _build_component_user_prompt(
        self,
        component: SSISDataFlowComponent,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build user prompt for component conversion."""

        prompt_parts = [
            f"## Convert SSIS {component.component_type}: {component.name}",
            "",
            "### Component Properties",
        ]

        # Add properties
        for key, value in component.properties.items():
            if value:
                prompt_parts.append(f"- **{key}**: {value[:200]}{'...' if len(str(value)) > 200 else ''}")

        # Add input columns
        if component.input_columns:
            prompt_parts.extend(["", "### Input Columns"])
            for col in component.input_columns[:10]:  # Limit columns
                prompt_parts.append(f"- {col.get('name', 'unknown')}")

        # Add output columns
        if component.output_columns:
            prompt_parts.extend(["", "### Output Columns"])
            for col in component.output_columns[:10]:
                col_info = f"- {col.get('name', 'unknown')}"
                if col.get('expression'):
                    col_info += f" = `{col['expression'][:100]}`"
                prompt_parts.append(col_info)

        # Add context
        if context:
            prompt_parts.extend(["", "### Context"])
            if context.get("input_dataframe"):
                prompt_parts.append(f"- Input DataFrame: `{context['input_dataframe']}`")
            if context.get("catalog"):
                prompt_parts.append(f"- Target Catalog: `{context['catalog']}`")

        # Add output requirements
        prompt_parts.extend([
            "",
            "### Required Output",
            "Generate PySpark code that:",
            "1. Implements the component's logic",
            "2. Uses the input DataFrame variable name if provided",
            "3. Outputs to a new DataFrame variable",
            "4. Includes comments explaining the transformation",
            "",
            "Output the code only, wrapped in ```python blocks."
        ])

        return "\n".join(prompt_parts)

    def _build_data_flow_user_prompt(
        self,
        data_flow: SSISDataFlow,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build user prompt for data flow conversion."""

        prompt_parts = [
            f"## Convert SSIS Data Flow: {data_flow.name}",
            "",
            "### Data Flow Components",
        ]

        # List components in order
        for i, comp in enumerate(data_flow.components, 1):
            prompt_parts.append(f"{i}. **{comp.name}** ({comp.component_type})")

        # Add path information
        if data_flow.paths:
            prompt_parts.extend(["", "### Data Paths"])
            for path in data_flow.paths[:10]:
                prompt_parts.append(f"- {path.name}: `{path.source_ref}` → `{path.dest_ref}`")

        # Add configuration context
        prompt_parts.extend([
            "",
            "### Configuration",
            f"- Target Catalog: `{self.config.default_catalog}`",
            f"- Target Schema: `{self.config.default_schema}`",
            f"- Use Delta Lake: `{self.config.delta_enabled}`",
            f"- Include source comments: `{self.config.include_source_comments}`",
        ])

        # Add output requirements
        prompt_parts.extend([
            "",
            "### Required Output",
            "Generate a complete PySpark notebook that:",
            "1. Implements all components in the correct order",
            "2. Maintains data lineage through the pipeline",
            "3. Uses Databricks notebook format (# COMMAND ----------)",
            "4. Includes setup cell with imports and configuration",
            "5. Handles errors appropriately",
            "6. Logs metrics (row counts, timings)",
            "",
            "Output the complete notebook code."
        ])

        return "\n".join(prompt_parts)

    def _build_workflow_user_prompt(
        self,
        package: SSISPackage,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build user prompt for workflow conversion."""

        prompt_parts = [
            f"## Convert SSIS Package to Databricks Workflow: {package.name}",
            "",
            f"**Description**: {package.description}",
            "",
            "### Connections",
        ]

        for conn in package.connections:
            prompt_parts.append(f"- **{conn.name}** ({conn.connection_type}): {conn.server or 'N/A'}")

        prompt_parts.extend(["", "### Variables"])
        for var in package.variables:
            prompt_parts.append(f"- **{var.name}** ({var.data_type}): {var.value}")

        prompt_parts.extend(["", "### Control Flow Tasks"])
        for task in package.tasks:
            self._add_task_to_prompt(task, prompt_parts, indent=0)

        if package.precedence_constraints:
            prompt_parts.extend(["", "### Precedence Constraints"])
            for pc in package.precedence_constraints:
                prompt_parts.append(f"- {pc.from_task} → {pc.to_task}")

        if package.event_handlers:
            prompt_parts.extend(["", "### Event Handlers"])
            for eh in package.event_handlers:
                prompt_parts.append(f"- **{eh.event_name}**: {len(eh.tasks)} tasks")

        prompt_parts.extend([
            "",
            "### Required Output",
            "Generate a Databricks Jobs API workflow definition (JSON) that:",
            "1. Maps each SSIS task to appropriate Databricks task type",
            "2. Preserves execution order via depends_on",
            "3. Converts variables to job parameters",
            "4. Configures email notifications for success/failure",
            "5. Includes appropriate job cluster configuration",
            "",
            "Output the complete workflow JSON."
        ])

        return "\n".join(prompt_parts)

    def _add_task_to_prompt(
        self,
        task: SSISTask,
        prompt_parts: List[str],
        indent: int = 0
    ):
        """Recursively add tasks to prompt."""
        prefix = "  " * indent
        prompt_parts.append(f"{prefix}- **{task.name}** ({task.task_type})")

        if task.data_flow:
            prompt_parts.append(f"{prefix}  - Data Flow: {len(task.data_flow.components)} components")

        for child in task.child_tasks:
            self._add_task_to_prompt(child, prompt_parts, indent + 1)

    def _format_examples(self, examples: List[ConversionExample]) -> List[Dict[str, str]]:
        """Format examples for few-shot learning."""
        formatted = []

        for ex in examples:
            formatted.append({
                "name": ex.name,
                "description": ex.description,
                "input": ex.input_content,
                "output": ex.output_content,
                "notes": ex.notes
            })

        return formatted

    def _get_context_tags(self, component: SSISDataFlowComponent) -> List[str]:
        """Extract context tags from component for example matching."""
        tags = [component.component_type.lower()]

        # Add tags based on component properties
        if "SqlCommand" in component.properties:
            tags.append("sql")
        if "lookup" in component.component_type.lower():
            tags.extend(["join", "dimension"])
        if "destination" in component.component_type.lower():
            tags.extend(["write", "output"])
        if "source" in component.component_type.lower():
            tags.extend(["read", "input"])

        return tags

    def get_full_prompt_text(self, prompt: ConversionPrompt) -> str:
        """Get the full prompt text for debugging/display."""
        parts = [
            "=== SYSTEM PROMPT ===",
            prompt.system_prompt,
            "",
            "=== EXAMPLES ===",
        ]

        for i, ex in enumerate(prompt.examples, 1):
            parts.extend([
                f"--- Example {i}: {ex['name']} ---",
                f"Input:\n{ex['input'][:500]}...",
                f"Output:\n{ex['output'][:500]}...",
                ""
            ])

        parts.extend([
            "=== USER PROMPT ===",
            prompt.user_prompt
        ])

        return "\n".join(parts)
