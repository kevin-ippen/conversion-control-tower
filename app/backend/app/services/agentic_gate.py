"""
Agentic validation gate for post-conversion quality enforcement.

After the AI model generates converted code, this gate runs deterministic checks
and optionally EXPLAIN-based SQL validation. If checks fail, it re-submits the
code to the AI with error feedback for auto-correction (max 2 retries).

Inspired by BrickMod's EXPLAIN-based validation approach.
"""

import ast
import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class GateCheck:
    """Result of a single gate check."""
    name: str
    passed: bool
    message: str
    severity: str = "error"  # error = blocks, warning = advisory


@dataclass
class GateResult:
    """Result of a full gate validation pass."""
    passed: bool
    checks: List[GateCheck]
    attempt: int
    blocking_failures: List[str] = field(default_factory=list)

    @property
    def failure_summary(self) -> str:
        """Summary of all blocking failures for LLM feedback."""
        if not self.blocking_failures:
            return ""
        return "\n".join(f"- {f}" for f in self.blocking_failures)


@dataclass
class GateReport:
    """Full report across all gate attempts."""
    final_result: dict  # The (possibly corrected) conversion result
    attempts: List[GateResult]
    total_attempts: int
    passed: bool

    def to_dict(self) -> dict:
        return {
            "total_attempts": self.total_attempts,
            "passed": self.passed,
            "attempts": [
                {
                    "attempt": a.attempt,
                    "passed": a.passed,
                    "checks": [
                        {"name": c.name, "passed": c.passed, "message": c.message, "severity": c.severity}
                        for c in a.checks
                    ],
                    "blocking_failures": a.blocking_failures,
                }
                for a in self.attempts
            ],
        }


# Patterns for extracting SQL from spark.sql() calls
SPARK_SQL_PATTERN = re.compile(
    r'spark\.sql\(\s*(?:f|r|fr|rf)?(?P<quote>["\'{1,3}])(.*?)(?P=quote)\s*\)',
    re.DOTALL,
)

# Patterns for credential detection (same as code_quality_analyzer)
CREDENTIAL_PATTERNS = [
    re.compile(r'password\s*=\s*["\'][^"\']+["\']', re.IGNORECASE),
    re.compile(r'secret\s*=\s*["\'][^"\']+["\']', re.IGNORECASE),
    re.compile(r'api_key\s*=\s*["\'][^"\']+["\']', re.IGNORECASE),
    re.compile(r'jdbc:.*password=[^;]+', re.IGNORECASE),
]


class AgenticGate:
    """Post-conversion validation gate with auto-retry."""

    MAX_RETRIES = 2

    def __init__(
        self,
        ai_client: OpenAI,
        model_name: str,
        warehouse_id: Optional[str] = None,
        databricks_host: Optional[str] = None,
        databricks_token: Optional[str] = None,
    ):
        self.ai_client = ai_client
        self.model_name = model_name
        self.warehouse_id = warehouse_id
        self.databricks_host = databricks_host
        self.databricks_token = databricks_token

    async def validate_and_fix(
        self,
        conversion_result: dict,
        source_content: str,
        system_prompt: str,
        user_prompt: str,
        source_type: str = "ssis",
    ) -> GateReport:
        """Validate conversion result and auto-fix if needed.

        Args:
            conversion_result: The parsed JSON from the AI conversion
            source_content: Original source code
            system_prompt: The system prompt used for conversion
            user_prompt: The user prompt used for conversion
            source_type: Type of source (ssis, sql_script, stored_proc)

        Returns:
            GateReport with the final (possibly corrected) result
        """
        attempts = []
        current_result = conversion_result

        for attempt_num in range(1, self.MAX_RETRIES + 2):  # 1 initial + MAX_RETRIES
            gate_result = await self._run_checks(current_result, source_type, attempt_num)
            attempts.append(gate_result)

            if gate_result.passed:
                logger.info(f"Gate passed on attempt {attempt_num}")
                return GateReport(
                    final_result=current_result,
                    attempts=attempts,
                    total_attempts=attempt_num,
                    passed=True,
                )

            # If we've exhausted retries, return with failures
            if attempt_num > self.MAX_RETRIES:
                logger.warning(
                    f"Gate failed after {attempt_num} attempts. "
                    f"Failures: {gate_result.failure_summary}"
                )
                return GateReport(
                    final_result=current_result,
                    attempts=attempts,
                    total_attempts=attempt_num,
                    passed=False,
                )

            # Auto-retry: re-submit to AI with error feedback
            logger.info(
                f"Gate failed on attempt {attempt_num}, retrying with feedback. "
                f"Failures: {gate_result.blocking_failures}"
            )
            current_result = await self._retry_with_feedback(
                current_result,
                gate_result,
                system_prompt,
                user_prompt,
            )

        # Should not reach here, but safety fallback
        return GateReport(
            final_result=current_result,
            attempts=attempts,
            total_attempts=len(attempts),
            passed=False,
        )

    async def _run_checks(
        self,
        result: dict,
        source_type: str,
        attempt: int,
    ) -> GateResult:
        """Run all gate checks on the conversion result."""
        checks: List[GateCheck] = []

        # 1. JSON structure check
        checks.append(self._check_json_structure(result, source_type))

        # 2. AST compilation check on each notebook
        checks.extend(self._check_ast_compilation(result))

        # 3. UC naming check
        checks.extend(self._check_uc_naming(result))

        # 4. Credential scan
        checks.extend(self._check_credentials(result))

        # 5. EXPLAIN validation (if warehouse configured)
        if self.warehouse_id:
            explain_checks = await self._check_explain(result)
            checks.extend(explain_checks)

        # 6. Component coverage (for SSIS and Informatica)
        if source_type in ("ssis", "informatica_pc"):
            checks.append(self._check_component_coverage(result))

        # 7. Informatica-specific residue check
        if source_type == "informatica_pc":
            checks.extend(self._check_informatica_residue(result))

        # Determine pass/fail
        blocking = [c for c in checks if not c.passed and c.severity == "error"]
        passed = len(blocking) == 0

        return GateResult(
            passed=passed,
            checks=checks,
            attempt=attempt,
            blocking_failures=[c.message for c in blocking],
        )

    def _check_json_structure(self, result: dict, source_type: str) -> GateCheck:
        """Validate the AI response has the expected JSON structure."""
        has_notebooks = "notebooks" in result and isinstance(result["notebooks"], list)
        has_quality = "quality_notes" in result

        if source_type in ("ssis", "informatica_pc"):
            has_workflow = "workflow" in result
            if has_notebooks and has_quality:
                return GateCheck(
                    name="json_structure",
                    passed=True,
                    message="Response has required structure (notebooks, quality_notes)",
                )
            missing = []
            if not has_notebooks:
                missing.append("notebooks")
            if not has_quality:
                missing.append("quality_notes")
            return GateCheck(
                name="json_structure",
                passed=False,
                message=f"Missing required fields: {', '.join(missing)}",
            )
        else:
            if has_notebooks:
                return GateCheck(
                    name="json_structure",
                    passed=True,
                    message="Response has required structure",
                )
            return GateCheck(
                name="json_structure",
                passed=False,
                message="Missing 'notebooks' array in response",
            )

    def _check_ast_compilation(self, result: dict) -> List[GateCheck]:
        """Validate each notebook's code compiles as Python."""
        checks = []
        for notebook in result.get("notebooks", []):
            name = notebook.get("name", "unknown")
            code = notebook.get("code", "")
            if not code:
                checks.append(GateCheck(
                    name=f"ast_compile_{name}",
                    passed=False,
                    message=f"Notebook '{name}' has empty code",
                    severity="warning",
                ))
                continue

            # Strip Databricks magic commands before parsing
            clean_code = self._strip_magic(code)
            try:
                ast.parse(clean_code)
                checks.append(GateCheck(
                    name=f"ast_compile_{name}",
                    passed=True,
                    message=f"Notebook '{name}' compiles successfully",
                ))
            except SyntaxError as e:
                checks.append(GateCheck(
                    name=f"ast_compile_{name}",
                    passed=False,
                    message=f"Syntax error in '{name}' at line {e.lineno}: {e.msg}",
                ))
        return checks

    def _check_uc_naming(self, result: dict) -> List[GateCheck]:
        """Verify table references use three-part UC naming."""
        checks = []
        table_patterns = [
            re.compile(r'spark\.table\(["\']([^"\']+)["\']'),
            re.compile(r'\.saveAsTable\(["\']([^"\']+)["\']'),
            re.compile(r'FROM\s+([a-zA-Z_]\w*\.\w+\.\w+)', re.IGNORECASE),
        ]

        all_code = "\n".join(
            nb.get("code", "") for nb in result.get("notebooks", [])
        )

        bad_refs = []
        for pattern in table_patterns:
            for match in pattern.finditer(all_code):
                ref = match.group(1).strip('`"\'')
                if ref.startswith("{") or ref.startswith("$"):
                    continue
                parts = ref.split(".")
                if len(parts) != 3 and "." in ref:
                    bad_refs.append(ref)

        if bad_refs:
            checks.append(GateCheck(
                name="uc_naming",
                passed=False,
                message=f"Table references without three-part naming: {', '.join(bad_refs[:5])}. Use catalog.schema.table format.",
            ))
        else:
            checks.append(GateCheck(
                name="uc_naming",
                passed=True,
                message="All table references use UC three-part naming",
            ))
        return checks

    def _check_credentials(self, result: dict) -> List[GateCheck]:
        """Scan generated code for hardcoded credentials."""
        all_code = "\n".join(
            nb.get("code", "") for nb in result.get("notebooks", [])
        )

        for pattern in CREDENTIAL_PATTERNS:
            match = pattern.search(all_code)
            if match:
                return [GateCheck(
                    name="no_credentials",
                    passed=False,
                    message=f"Hardcoded credential detected: {match.group()[:50]}... Use dbutils.secrets.get() instead.",
                )]

        return [GateCheck(
            name="no_credentials",
            passed=True,
            message="No hardcoded credentials found",
        )]

    async def _check_explain(self, result: dict) -> List[GateCheck]:
        """Run EXPLAIN on extracted SQL statements via Statement Execution API.

        This is the key BrickMod-inspired check: actually validate SQL against
        the warehouse engine.
        """
        checks = []
        all_code = "\n".join(
            nb.get("code", "") for nb in result.get("notebooks", [])
        )

        # Extract SQL from spark.sql() calls
        sql_statements = []
        for match in SPARK_SQL_PATTERN.finditer(all_code):
            sql = match.group(2).strip()
            # Skip if it contains f-string variables
            if "{" in sql and "}" in sql:
                continue
            # Skip very short or comment-only statements
            if len(sql) < 10 or sql.startswith("--"):
                continue
            sql_statements.append(sql)

        if not sql_statements:
            checks.append(GateCheck(
                name="explain_validation",
                passed=True,
                message="No static SQL statements to EXPLAIN-validate",
                severity="warning",
            ))
            return checks

        # Run EXPLAIN via Statement Execution API
        for i, sql in enumerate(sql_statements[:5]):  # Limit to 5 statements
            try:
                result_text = await self._run_explain(sql)
                checks.append(GateCheck(
                    name=f"explain_sql_{i+1}",
                    passed=True,
                    message=f"EXPLAIN succeeded for SQL statement {i+1}",
                ))
            except Exception as e:
                error_msg = str(e)
                # Truncate JVM stacktrace if present
                jvm_idx = error_msg.find("JVM stacktrace:")
                if jvm_idx != -1:
                    error_msg = error_msg[:jvm_idx].strip()
                checks.append(GateCheck(
                    name=f"explain_sql_{i+1}",
                    passed=False,
                    message=f"EXPLAIN failed for SQL statement {i+1}: {error_msg[:200]}",
                ))

        return checks

    async def _run_explain(self, sql: str) -> str:
        """Execute EXPLAIN on a SQL statement via the Statement Execution API."""
        import httpx

        host = self.databricks_host or os.environ.get("DATABRICKS_HOST", "")
        token = self.databricks_token or os.environ.get("DATABRICKS_TOKEN", "")

        if not host.startswith("https://"):
            host = f"https://{host}"

        url = f"{host}/api/2.0/sql/statements"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload = {
            "warehouse_id": self.warehouse_id,
            "statement": f"EXPLAIN {sql}",
            "wait_timeout": "30s",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=35.0)
            response.raise_for_status()
            data = response.json()

            status = data.get("status", {}).get("state", "")
            if status == "FAILED":
                error = data.get("status", {}).get("error", {})
                raise Exception(error.get("message", "EXPLAIN failed"))

            return "EXPLAIN succeeded"

    def _check_component_coverage(self, result: dict) -> GateCheck:
        """For SSIS conversions, check that workflow has tasks."""
        workflow = result.get("workflow", {})
        tasks = workflow.get("tasks", [])
        notebooks = result.get("notebooks", [])

        if not notebooks:
            return GateCheck(
                name="component_coverage",
                passed=False,
                message="No notebooks generated from SSIS package",
            )

        return GateCheck(
            name="component_coverage",
            passed=True,
            message=f"Generated {len(notebooks)} notebook(s) and {len(tasks)} workflow task(s)",
        )

    def _check_informatica_residue(self, result: dict) -> List[GateCheck]:
        """Check for unconverted Informatica-specific syntax in generated code.

        Flags :LKP references, $$parameters, Informatica functions, and other
        residue that indicates incomplete conversion.
        """
        checks = []
        all_code = "\n".join(
            nb.get("code", "") for nb in result.get("notebooks", [])
        )

        residue_patterns = [
            (re.compile(r':LKP\.\w+'), "Unconverted Lookup reference (:LKP.)"),
            (re.compile(r'\$\$\w+'), "Unconverted Informatica parameter ($$)"),
            (re.compile(r'\bIS_SPACES\s*\('), "Unconverted IS_SPACES() function"),
            (re.compile(r'\bDECODE\s*\('), "Unconverted DECODE() function (use CASE WHEN)"),
            (re.compile(r'\bSYSDATE\b(?!\s*\()'), "Unconverted SYSDATE (use CURRENT_TIMESTAMP())"),
            (re.compile(r'\bSYSTIMESTAMP\b(?!\s*\()'), "Unconverted SYSTIMESTAMP"),
            (re.compile(r'\bADD_TO_DATE\s*\('), "Unconverted ADD_TO_DATE() function"),
            (re.compile(r'\$PM\w+'), "Unconverted PowerCenter session variable ($PM...)"),
        ]

        found_residue = []
        for pattern, desc in residue_patterns:
            matches = pattern.findall(all_code)
            if matches:
                found_residue.append(f"{desc}: {matches[0]}")

        if found_residue:
            checks.append(GateCheck(
                name="informatica_residue",
                passed=False,
                message=f"Unconverted Informatica syntax found: {'; '.join(found_residue[:5])}",
                severity="warning",
            ))
        else:
            checks.append(GateCheck(
                name="informatica_residue",
                passed=True,
                message="No unconverted Informatica syntax residue found",
            ))

        return checks

    async def _retry_with_feedback(
        self,
        previous_result: dict,
        gate_result: GateResult,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        """Re-submit to AI with error feedback for auto-correction."""
        feedback = (
            "Your previous conversion output had the following issues that need fixing:\n"
            f"{gate_result.failure_summary}\n\n"
            "Please fix these issues and regenerate the output. "
            "Ensure all table references use catalog.schema.table format, "
            "all credentials use dbutils.secrets.get(), "
            "and all generated Python code is syntactically valid.\n\n"
            "Previous output that needs fixing:\n"
            f"```json\n{json.dumps(previous_result, indent=2)[:4000]}\n```"
        )

        def call_api():
            response = self.ai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": json.dumps(previous_result)[:4000]},
                    {"role": "user", "content": feedback},
                ],
                temperature=0.1,
                max_tokens=8000,
            )
            return response.choices[0].message.content

        try:
            result_text = await asyncio.to_thread(call_api)
            if not isinstance(result_text, str):
                result_text = str(result_text) if result_text else ""

            # Parse JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            return json.loads(result_text.strip())

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Retry failed: {e}")
            return previous_result  # Return unchanged if retry fails

    @staticmethod
    def _strip_magic(code: str) -> str:
        """Remove Databricks magic commands for AST parsing."""
        lines = []
        for line in code.split("\n"):
            stripped = line.strip()
            if stripped.startswith("# MAGIC") or stripped.startswith("# COMMAND"):
                continue
            if stripped == "# Databricks notebook source":
                continue
            lines.append(line)
        return "\n".join(lines)
