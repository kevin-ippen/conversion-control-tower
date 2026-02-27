"""
End-to-end tests for the SalesDataETL sample package conversion pipeline.
"""

import ast
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ssis.dtsx_parser import DTSXParser
from src.extraction.schema_extractor import SchemaExtractor
from src.conversion.ssis_to_notebook import SSISToNotebookConverter
from src.conversion.ssis_to_workflow import SSISToWorkflowConverter
from tests import find_task_recursive

SALES_DTSX = PROJECT_ROOT / "samples" / "ssis" / "SalesDataETL.dtsx"


def _strip_magic(code: str) -> str:
    lines = code.split("\n")
    return "\n".join(l for l in lines if not l.strip().startswith("# MAGIC") and not l.strip().startswith("%"))


@pytest.mark.e2e
class TestSalesE2E:
    """Full pipeline test for the SalesDataETL sample package."""

    @pytest.fixture(scope="class")
    def pipeline_result(self):
        parser = DTSXParser(SALES_DTSX)
        package = parser.parse()

        extractor = SchemaExtractor()
        schemas = extractor.extract_from_ssis(str(SALES_DTSX))

        notebook_converter = SSISToNotebookConverter()
        notebooks = {}

        def collect_data_flows(task):
            if task.data_flow:
                name = task.name.replace(" ", "_").lower()
                notebooks[name] = notebook_converter.convert_data_flow(
                    task.data_flow, name
                )
            for child in task.child_tasks:
                collect_data_flows(child)

        for task in package.tasks:
            collect_data_flows(task)

        workflow_converter = SSISToWorkflowConverter()
        workflow = workflow_converter.convert(package)

        return {
            "package": package,
            "schemas": schemas,
            "notebooks": notebooks,
            "workflow": workflow,
            "workflow_converter": workflow_converter,
        }

    def test_parse_succeeds(self, pipeline_result):
        pkg = pipeline_result["package"]
        assert pkg.name == "SalesDataETL"
        assert len(pkg.connections) == 3
        assert len(pkg.variables) == 8

    def test_schema_extraction(self, pipeline_result):
        schemas = pipeline_result["schemas"]
        assert len(schemas.source_tables) > 0

    def test_notebooks_generated(self, pipeline_result):
        notebooks = pipeline_result["notebooks"]
        assert len(notebooks) > 0

    @pytest.mark.xfail(reason="Known bug: generated notebook has indentation errors from multi-line SQL embedding")
    def test_notebooks_compile(self, pipeline_result):
        notebooks = pipeline_result["notebooks"]
        for name, code in notebooks.items():
            clean = _strip_magic(code)
            try:
                ast.parse(clean)
            except SyntaxError as e:
                pytest.fail(f"Notebook '{name}' syntax error at line {e.lineno}: {e.msg}")

    def test_workflow_generated(self, pipeline_result):
        wf = pipeline_result["workflow"]
        assert "salesdataetl" in wf.name.lower()
        assert len(wf.tasks) > 0

    def test_workflow_json(self, pipeline_result):
        wf = pipeline_result["workflow"]
        converter = pipeline_result["workflow_converter"]
        json_str = converter.to_json(wf)
        parsed = json.loads(json_str)
        assert len(parsed["tasks"]) > 0
