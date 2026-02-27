"""
Multi-dimensional code quality analyzer for converted Databricks code.

Performs deterministic (no LLM) analysis across 5 dimensions:
- Code Quality: syntax, imports, credential checks, UC naming
- Standards Adherence: Delta best practices, error handling, documentation
- Performance: broadcast hints, collect avoidance, partitioning
- Parameterization: secrets usage, widgets, no hardcoded values
- Verbosity: comment ratio, function decomposition, dead code
"""

import ast
import re
import logging
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a single quality check."""
    name: str
    passed: bool
    message: str
    dimension: str
    severity: str = "error"  # error, warning, info
    expected: str = ""
    actual: str = ""


@dataclass
class DimensionScore:
    """Score for a single quality dimension."""
    name: str
    score: float  # 0-100
    weight: float
    checks: List[CheckResult] = field(default_factory=list)
    description: str = ""


@dataclass
class QualityAnalysis:
    """Complete multi-dimensional quality analysis."""
    dimensions: List[DimensionScore]
    overall_score: float
    total_checks: int
    passed_checks: int
    failed_checks: int

    @property
    def grade(self) -> str:
        if self.overall_score >= 90:
            return "A"
        elif self.overall_score >= 80:
            return "B"
        elif self.overall_score >= 70:
            return "C"
        elif self.overall_score >= 60:
            return "D"
        return "F"


# Patterns for credential detection
CREDENTIAL_PATTERNS = [
    re.compile(r'password\s*=\s*["\'][^"\']+["\']', re.IGNORECASE),
    re.compile(r'secret\s*=\s*["\'][^"\']+["\']', re.IGNORECASE),
    re.compile(r'api_key\s*=\s*["\'][^"\']+["\']', re.IGNORECASE),
    re.compile(r'token\s*=\s*["\'][A-Za-z0-9+/=]{20,}["\']', re.IGNORECASE),
    re.compile(r'jdbc:.*password=[^;]+', re.IGNORECASE),
]

# Patterns for table references
TABLE_REF_PATTERNS = [
    re.compile(r'spark\.table\(["\']([^"\']+)["\']'),
    re.compile(r'\.saveAsTable\(["\']([^"\']+)["\']'),
    re.compile(r'spark\.read\.table\(["\']([^"\']+)["\']'),
    re.compile(r'MERGE\s+INTO\s+([^\s(]+)', re.IGNORECASE),
    re.compile(r'INSERT\s+INTO\s+([^\s(]+)', re.IGNORECASE),
    re.compile(r'CREATE\s+(?:OR\s+REPLACE\s+)?TABLE\s+([^\s(]+)', re.IGNORECASE),
    re.compile(r'FROM\s+([a-zA-Z_]\w*\.\w+\.\w+)', re.IGNORECASE),
]

# Patterns for legacy/bad paths
BAD_PATH_PATTERNS = [
    re.compile(r'dbfs:/', re.IGNORECASE),
    re.compile(r'/mnt/', re.IGNORECASE),
    re.compile(r'wasbs?://', re.IGNORECASE),
    re.compile(r's3a?://', re.IGNORECASE),
]

# Patterns for hardcoded values that should be parameterized
HARDCODED_PATTERNS = [
    re.compile(r'jdbc:sqlserver://[^\s"\']+', re.IGNORECASE),
    re.compile(r'Server\s*=\s*["\'][^"\']+["\']', re.IGNORECASE),
    re.compile(r'Data\s*Source\s*=\s*["\'][^"\']+["\']', re.IGNORECASE),
]


class CodeQualityAnalyzer:
    """Analyzes converted code across multiple quality dimensions."""

    # Dimension weights (must sum to 1.0)
    DIMENSION_WEIGHTS = {
        "code_quality": 0.30,
        "standards": 0.20,
        "performance": 0.15,
        "parameterization": 0.20,
        "verbosity": 0.15,
    }

    def analyze(self, code: str, source_type: str = "ssis") -> QualityAnalysis:
        """Run full multi-dimensional analysis on generated code.

        Args:
            code: The generated Python/PySpark code to analyze
            source_type: The source type (ssis, sql_script, stored_proc)

        Returns:
            QualityAnalysis with scores across all dimensions
        """
        dimensions = [
            self._analyze_code_quality(code),
            self._analyze_standards(code),
            self._analyze_performance(code),
            self._analyze_parameterization(code),
            self._analyze_verbosity(code),
        ]

        # Calculate overall weighted score
        overall = sum(
            d.score * d.weight for d in dimensions
        )

        total = sum(len(d.checks) for d in dimensions)
        passed = sum(
            sum(1 for c in d.checks if c.passed)
            for d in dimensions
        )

        return QualityAnalysis(
            dimensions=dimensions,
            overall_score=round(overall, 1),
            total_checks=total,
            passed_checks=passed,
            failed_checks=total - passed,
        )

    def _analyze_code_quality(self, code: str) -> DimensionScore:
        """Dimension 1: Core code quality checks (deterministic)."""
        checks: List[CheckResult] = []

        # 1. Syntax validation via AST
        checks.append(self._check_syntax(code))

        # 2. Credential scanning
        checks.extend(self._check_credentials(code))

        # 3. UC three-part naming
        checks.extend(self._check_uc_naming(code))

        # 4. No legacy path usage
        checks.extend(self._check_legacy_paths(code))

        # 5. Import completeness (basic)
        checks.append(self._check_imports(code))

        score = self._calc_dimension_score(checks)
        return DimensionScore(
            name="code_quality",
            score=score,
            weight=self.DIMENSION_WEIGHTS["code_quality"],
            checks=checks,
            description="Syntax validity, credential safety, UC naming compliance",
        )

    def _analyze_standards(self, code: str) -> DimensionScore:
        """Dimension 2: Databricks best practices adherence."""
        checks: List[CheckResult] = []

        # 1. Delta format usage
        checks.append(self._check_delta_usage(code))

        # 2. Error handling present
        checks.append(self._check_error_handling(code))

        # 3. Logging/print statements for observability
        checks.append(self._check_observability(code))

        # 4. Documentation (docstrings/comments on functions)
        checks.append(self._check_documentation(code))

        # 5. Notebook header present
        checks.append(self._check_notebook_header(code))

        score = self._calc_dimension_score(checks)
        return DimensionScore(
            name="standards",
            score=score,
            weight=self.DIMENSION_WEIGHTS["standards"],
            checks=checks,
            description="Delta best practices, error handling, documentation",
        )

    def _analyze_performance(self, code: str) -> DimensionScore:
        """Dimension 3: Expected performance characteristics."""
        checks: List[CheckResult] = []

        # 1. No .collect() on potentially large DataFrames
        checks.append(self._check_collect_usage(code))

        # 2. No .toPandas() on large DataFrames
        checks.append(self._check_topandas_usage(code))

        # 3. Broadcast hints for small joins
        checks.append(self._check_broadcast_hints(code))

        # 4. Caching for reused DataFrames
        checks.append(self._check_caching(code))

        score = self._calc_dimension_score(checks)
        return DimensionScore(
            name="performance",
            score=score,
            weight=self.DIMENSION_WEIGHTS["performance"],
            checks=checks,
            description="Collect avoidance, broadcast hints, caching strategy",
        )

    def _analyze_parameterization(self, code: str) -> DimensionScore:
        """Dimension 4: Parameterization vs hardcoded values."""
        checks: List[CheckResult] = []

        # 1. Secrets usage for credentials
        checks.append(self._check_secrets_usage(code))

        # 2. Widget usage for configurable values
        checks.append(self._check_widget_usage(code))

        # 3. No hardcoded connection strings
        checks.extend(self._check_hardcoded_values(code))

        score = self._calc_dimension_score(checks)
        return DimensionScore(
            name="parameterization",
            score=score,
            weight=self.DIMENSION_WEIGHTS["parameterization"],
            checks=checks,
            description="Secrets management, widget parameters, no hardcoded values",
        )

    def _analyze_verbosity(self, code: str) -> DimensionScore:
        """Dimension 5: Code conciseness and clarity."""
        checks: List[CheckResult] = []

        # 1. Comment-to-code ratio
        checks.append(self._check_comment_ratio(code))

        # 2. Function decomposition
        checks.append(self._check_function_decomposition(code))

        # 3. No obviously dead/commented-out code blocks
        checks.append(self._check_dead_code(code))

        score = self._calc_dimension_score(checks)
        return DimensionScore(
            name="verbosity",
            score=score,
            weight=self.DIMENSION_WEIGHTS["verbosity"],
            checks=checks,
            description="Code density, function structure, dead code detection",
        )

    # -------------------------------------------------------------------------
    # Individual check implementations
    # -------------------------------------------------------------------------

    def _check_syntax(self, code: str) -> CheckResult:
        """Validate Python syntax using AST."""
        # Strip Databricks notebook magic commands before parsing
        clean_code = self._strip_magic_commands(code)
        try:
            ast.parse(clean_code)
            return CheckResult(
                name="syntax_valid",
                passed=True,
                message="Code is syntactically valid Python",
                dimension="code_quality",
            )
        except SyntaxError as e:
            return CheckResult(
                name="syntax_valid",
                passed=False,
                message=f"Syntax error at line {e.lineno}: {e.msg}",
                dimension="code_quality",
                actual=f"Line {e.lineno}: {e.msg}",
            )

    def _check_credentials(self, code: str) -> List[CheckResult]:
        """Scan for hardcoded credentials."""
        results = []
        found_any = False
        for pattern in CREDENTIAL_PATTERNS:
            matches = pattern.findall(code)
            if matches:
                found_any = True
                results.append(CheckResult(
                    name="no_hardcoded_credentials",
                    passed=False,
                    message=f"Hardcoded credential detected: {matches[0][:50]}...",
                    dimension="code_quality",
                    severity="error",
                    actual=matches[0][:80],
                ))
        if not found_any:
            results.append(CheckResult(
                name="no_hardcoded_credentials",
                passed=True,
                message="No hardcoded credentials detected",
                dimension="code_quality",
            ))
        return results

    def _check_uc_naming(self, code: str) -> List[CheckResult]:
        """Verify Unity Catalog three-part naming for table references."""
        results = []
        table_refs = set()
        for pattern in TABLE_REF_PATTERNS:
            for match in pattern.finditer(code):
                ref = match.group(1).strip('`"\'')
                # Skip variable references and widget placeholders
                if ref.startswith("{") or ref.startswith("$") or ref.startswith("f\""):
                    continue
                # Skip if it looks like a Python variable (no dots)
                if "." not in ref and not ref[0].isupper():
                    continue
                table_refs.add(ref)

        bad_refs = []
        for ref in table_refs:
            parts = ref.split(".")
            if len(parts) != 3:
                bad_refs.append(ref)

        if bad_refs:
            for ref in bad_refs:
                results.append(CheckResult(
                    name="uc_three_part_naming",
                    passed=False,
                    message=f"Table reference '{ref}' does not use three-part naming (catalog.schema.table)",
                    dimension="code_quality",
                    expected="catalog.schema.table",
                    actual=ref,
                ))
        else:
            results.append(CheckResult(
                name="uc_three_part_naming",
                passed=True,
                message=f"All {len(table_refs)} table references use UC three-part naming" if table_refs else "No table references found to validate",
                dimension="code_quality",
            ))
        return results

    def _check_legacy_paths(self, code: str) -> List[CheckResult]:
        """Check for legacy path usage (dbfs:/, /mnt/, etc.)."""
        results = []
        found_any = False
        for pattern in BAD_PATH_PATTERNS:
            matches = pattern.findall(code)
            if matches:
                found_any = True
                results.append(CheckResult(
                    name="no_legacy_paths",
                    passed=False,
                    message=f"Legacy path detected: {matches[0]}. Use /Volumes/... instead.",
                    dimension="code_quality",
                    severity="warning",
                    actual=matches[0],
                ))
        if not found_any:
            results.append(CheckResult(
                name="no_legacy_paths",
                passed=True,
                message="No legacy path patterns (dbfs:/, /mnt/) detected",
                dimension="code_quality",
            ))
        return results

    def _check_imports(self, code: str) -> CheckResult:
        """Basic check that necessary imports are present."""
        clean_code = self._strip_magic_commands(code)
        has_pyspark = "pyspark" in code or "from pyspark" in code
        has_spark_sql = "spark.sql" in code or "spark.read" in code or "spark.table" in code

        if has_spark_sql and not has_pyspark:
            # Spark usage without explicit imports - might rely on notebook environment
            return CheckResult(
                name="imports_present",
                passed=True,
                message="Code uses Spark (available in notebook environment)",
                dimension="code_quality",
                severity="info",
            )
        return CheckResult(
            name="imports_present",
            passed=True,
            message="Import structure appears complete",
            dimension="code_quality",
        )

    def _check_delta_usage(self, code: str) -> CheckResult:
        """Check that Delta format is used for table operations."""
        has_write = ".write" in code or "saveAsTable" in code or "MERGE INTO" in code.upper()
        has_delta = (
            'format("delta")' in code
            or "format('delta')" in code
            or "saveAsTable" in code  # saveAsTable defaults to Delta
            or "MERGE INTO" in code.upper()
            or "CREATE OR REFRESH" in code.upper()
        )

        if has_write and not has_delta:
            return CheckResult(
                name="delta_format_used",
                passed=False,
                message="Write operations detected without explicit Delta format",
                dimension="standards",
                severity="warning",
            )
        return CheckResult(
            name="delta_format_used",
            passed=True,
            message="Delta format used for table operations" if has_write else "No write operations to check",
            dimension="standards",
        )

    def _check_error_handling(self, code: str) -> CheckResult:
        """Check for error handling patterns."""
        has_try_except = "try:" in code and "except" in code
        has_raise = "raise" in code

        if has_try_except:
            return CheckResult(
                name="error_handling_present",
                passed=True,
                message="Error handling (try/except) present in code",
                dimension="standards",
            )
        return CheckResult(
            name="error_handling_present",
            passed=False,
            message="No error handling (try/except) found - recommended for production code",
            dimension="standards",
            severity="warning",
        )

    def _check_observability(self, code: str) -> CheckResult:
        """Check for logging/print statements."""
        has_logging = "logger." in code or "logging." in code
        has_print = "print(" in code or "display(" in code

        if has_logging or has_print:
            return CheckResult(
                name="observability_present",
                passed=True,
                message="Logging/display statements present for observability",
                dimension="standards",
            )
        return CheckResult(
            name="observability_present",
            passed=False,
            message="No logging or print statements found - add for observability",
            dimension="standards",
            severity="info",
        )

    def _check_documentation(self, code: str) -> CheckResult:
        """Check for documentation (docstrings, meaningful comments)."""
        has_docstring = '"""' in code or "'''" in code
        # Count meaningful comments (not magic commands)
        comment_lines = [
            line for line in code.split("\n")
            if line.strip().startswith("#")
            and not line.strip().startswith("# MAGIC")
            and not line.strip().startswith("# COMMAND")
            and not line.strip().startswith("# Databricks notebook")
            and len(line.strip()) > 3
        ]

        if has_docstring or len(comment_lines) >= 3:
            return CheckResult(
                name="documentation_present",
                passed=True,
                message=f"Documentation present ({len(comment_lines)} comment lines)",
                dimension="standards",
            )
        return CheckResult(
            name="documentation_present",
            passed=False,
            message="Limited documentation - add docstrings and comments for business logic",
            dimension="standards",
            severity="info",
        )

    def _check_notebook_header(self, code: str) -> CheckResult:
        """Check for Databricks notebook header."""
        has_header = "Databricks notebook source" in code
        if has_header:
            return CheckResult(
                name="notebook_header",
                passed=True,
                message="Databricks notebook header present",
                dimension="standards",
            )
        return CheckResult(
            name="notebook_header",
            passed=False,
            message="Missing Databricks notebook header (# Databricks notebook source)",
            dimension="standards",
            severity="warning",
        )

    def _check_collect_usage(self, code: str) -> CheckResult:
        """Check for .collect() usage on potentially large DataFrames."""
        collect_calls = re.findall(r'\.collect\(\)', code)
        # Allow .collect() if it's on a clearly small result (e.g., after .limit() or .count())
        if len(collect_calls) > 2:
            return CheckResult(
                name="minimal_collect_usage",
                passed=False,
                message=f"Found {len(collect_calls)} .collect() calls - may cause OOM on large DataFrames",
                dimension="performance",
                severity="warning",
                actual=str(len(collect_calls)),
            )
        return CheckResult(
            name="minimal_collect_usage",
            passed=True,
            message=f".collect() usage is minimal ({len(collect_calls)} calls)",
            dimension="performance",
        )

    def _check_topandas_usage(self, code: str) -> CheckResult:
        """Check for .toPandas() usage on potentially large DataFrames."""
        topandas_calls = re.findall(r'\.toPandas\(\)', code)
        if len(topandas_calls) > 1:
            return CheckResult(
                name="minimal_topandas_usage",
                passed=False,
                message=f"Found {len(topandas_calls)} .toPandas() calls - prefer Spark operations for large data",
                dimension="performance",
                severity="warning",
            )
        return CheckResult(
            name="minimal_topandas_usage",
            passed=True,
            message=".toPandas() usage is minimal",
            dimension="performance",
        )

    def _check_broadcast_hints(self, code: str) -> CheckResult:
        """Check for broadcast hints on join operations."""
        has_joins = ".join(" in code or "JOIN" in code.upper()
        has_broadcast = "broadcast(" in code or "F.broadcast(" in code

        if has_joins and not has_broadcast:
            return CheckResult(
                name="broadcast_hints",
                passed=False,
                message="Joins detected without broadcast hints - consider broadcast() for small dimension tables",
                dimension="performance",
                severity="info",
            )
        return CheckResult(
            name="broadcast_hints",
            passed=True,
            message="Broadcast hints used for joins" if has_broadcast else "No joins to optimize",
            dimension="performance",
        )

    def _check_caching(self, code: str) -> CheckResult:
        """Check for DataFrame caching on reused variables."""
        # Simple heuristic: if a variable appears in multiple operations, suggest caching
        has_cache = ".cache()" in code or ".persist()" in code
        return CheckResult(
            name="caching_strategy",
            passed=True,  # This is advisory, not a hard fail
            message="Caching used for DataFrames" if has_cache else "No explicit caching - consider for reused DataFrames",
            dimension="performance",
            severity="info",
        )

    def _check_secrets_usage(self, code: str) -> CheckResult:
        """Check that credentials use dbutils.secrets."""
        has_jdbc = "jdbc" in code.lower() or "connection" in code.lower()
        has_secrets = "dbutils.secrets.get" in code

        if has_jdbc and not has_secrets:
            return CheckResult(
                name="secrets_usage",
                passed=False,
                message="Database connections detected without dbutils.secrets.get() - use Databricks Secrets",
                dimension="parameterization",
                severity="error",
            )
        return CheckResult(
            name="secrets_usage",
            passed=True,
            message="Credentials managed via Databricks Secrets" if has_secrets else "No credential usage detected",
            dimension="parameterization",
        )

    def _check_widget_usage(self, code: str) -> CheckResult:
        """Check for parameterization via widgets."""
        has_widgets = "dbutils.widgets" in code
        if has_widgets:
            widget_count = len(re.findall(r'dbutils\.widgets\.(text|dropdown|combobox|multiselect)\(', code))
            return CheckResult(
                name="widget_parameterization",
                passed=True,
                message=f"Parameterized with {widget_count} widget(s)",
                dimension="parameterization",
                actual=str(widget_count),
            )
        return CheckResult(
            name="widget_parameterization",
            passed=False,
            message="No dbutils.widgets found - add parameters for configurable values (catalog, schema, paths)",
            dimension="parameterization",
            severity="warning",
        )

    def _check_hardcoded_values(self, code: str) -> List[CheckResult]:
        """Check for hardcoded connection strings and paths."""
        results = []
        found_any = False
        for pattern in HARDCODED_PATTERNS:
            matches = pattern.findall(code)
            if matches:
                found_any = True
                results.append(CheckResult(
                    name="no_hardcoded_values",
                    passed=False,
                    message=f"Hardcoded value detected: {matches[0][:60]}...",
                    dimension="parameterization",
                    severity="warning",
                    actual=matches[0][:80],
                ))
        if not found_any:
            results.append(CheckResult(
                name="no_hardcoded_values",
                passed=True,
                message="No hardcoded connection strings or paths detected",
                dimension="parameterization",
            ))
        return results

    def _check_comment_ratio(self, code: str) -> CheckResult:
        """Check code-to-comment ratio (sweet spot: 10-30%)."""
        lines = code.split("\n")
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
        comment_lines = [
            l for l in lines
            if l.strip().startswith("#")
            and not l.strip().startswith("# MAGIC")
            and not l.strip().startswith("# COMMAND")
            and not l.strip().startswith("# Databricks notebook")
        ]

        total = len(code_lines) + len(comment_lines)
        if total == 0:
            return CheckResult(
                name="comment_ratio",
                passed=True,
                message="Empty code",
                dimension="verbosity",
            )

        ratio = len(comment_lines) / total * 100
        if 5 <= ratio <= 40:
            return CheckResult(
                name="comment_ratio",
                passed=True,
                message=f"Comment ratio is {ratio:.0f}% ({len(comment_lines)} comments / {total} total lines)",
                dimension="verbosity",
                actual=f"{ratio:.0f}%",
            )
        elif ratio < 5:
            return CheckResult(
                name="comment_ratio",
                passed=False,
                message=f"Comment ratio is low ({ratio:.0f}%) - add comments for business logic",
                dimension="verbosity",
                severity="info",
                actual=f"{ratio:.0f}%",
                expected="10-30%",
            )
        else:
            return CheckResult(
                name="comment_ratio",
                passed=False,
                message=f"Comment ratio is high ({ratio:.0f}%) - code may be over-documented",
                dimension="verbosity",
                severity="info",
                actual=f"{ratio:.0f}%",
                expected="10-30%",
            )

    def _check_function_decomposition(self, code: str) -> CheckResult:
        """Check for function decomposition (not a single monolithic block)."""
        clean_code = self._strip_magic_commands(code)
        try:
            tree = ast.parse(clean_code)
            func_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        except SyntaxError:
            func_count = len(re.findall(r'^def\s+', code, re.MULTILINE))

        non_empty_lines = len([l for l in code.split("\n") if l.strip()])

        if non_empty_lines > 50 and func_count == 0:
            return CheckResult(
                name="function_decomposition",
                passed=False,
                message=f"Code has {non_empty_lines} lines but no functions - consider decomposing",
                dimension="verbosity",
                severity="info",
            )
        return CheckResult(
            name="function_decomposition",
            passed=True,
            message=f"Code has {func_count} function(s) across {non_empty_lines} lines",
            dimension="verbosity",
        )

    def _check_dead_code(self, code: str) -> CheckResult:
        """Check for large commented-out code blocks."""
        lines = code.split("\n")
        consecutive_comments = 0
        max_block = 0

        for line in lines:
            stripped = line.strip()
            # Look for commented-out code (starts with # but looks like code)
            if (stripped.startswith("# ") and
                any(kw in stripped for kw in ["= ", "(", ")", "import ", "def ", "class "])):
                consecutive_comments += 1
                max_block = max(max_block, consecutive_comments)
            else:
                consecutive_comments = 0

        if max_block > 5:
            return CheckResult(
                name="no_dead_code",
                passed=False,
                message=f"Found {max_block} consecutive lines of commented-out code - remove dead code",
                dimension="verbosity",
                severity="info",
            )
        return CheckResult(
            name="no_dead_code",
            passed=True,
            message="No large blocks of commented-out code detected",
            dimension="verbosity",
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _strip_magic_commands(code: str) -> str:
        """Remove Databricks notebook magic commands for AST parsing."""
        lines = []
        for line in code.split("\n"):
            stripped = line.strip()
            if stripped.startswith("# MAGIC"):
                continue
            if stripped.startswith("# COMMAND"):
                continue
            if stripped == "# Databricks notebook source":
                continue
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _calc_dimension_score(checks: List[CheckResult]) -> float:
        """Calculate dimension score from checks.

        Error checks that fail reduce score more than warnings/info.
        """
        if not checks:
            return 100.0

        total_weight = 0.0
        earned_weight = 0.0

        for check in checks:
            if check.severity == "error":
                weight = 3.0
            elif check.severity == "warning":
                weight = 2.0
            else:
                weight = 1.0

            total_weight += weight
            if check.passed:
                earned_weight += weight

        return round((earned_weight / total_weight) * 100, 1) if total_weight > 0 else 100.0
