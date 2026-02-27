"""
In-process converter service using FMAPI.

Runs conversion logic directly in the app process instead of submitting
Databricks Jobs, since the Jobs API scope isn't available to Databricks Apps.

Uses PromptBuilder for config-driven prompts and AgenticGate for post-conversion
validation with auto-retry.
"""

import asyncio
import logging
import os
import json
from typing import Optional
from datetime import datetime

from openai import OpenAI

from ..config import Settings
from ..models.conversion import ConversionJob, ConversionStatus, SourceType, AIModel, OutputFormat
from .tracker import get_tracker
from .volume_manager import VolumeManager
from .prompt_builder import PromptBuilder
from .agentic_gate import AgenticGate

logger = logging.getLogger(__name__)


class ConverterService:
    """Runs conversions in-process using FMAPI."""

    def __init__(self, settings: Settings, volume_manager: VolumeManager):
        self.settings = settings
        self.volume_manager = volume_manager
        self.tracker = get_tracker()
        self.prompt_builder = PromptBuilder()

        # Initialize OpenAI client for FMAPI
        self._host = os.environ.get("DATABRICKS_HOST", "")
        if not self._host.startswith("https://"):
            self._host = f"https://{self._host}"

        # Get token from environment (SP or OBO)
        self._token = os.environ.get("DATABRICKS_TOKEN", "")
        if not self._token:
            # Try to get from SDK's auth
            from databricks.sdk import WorkspaceClient
            client = WorkspaceClient(host=self._host)
            self._token = client.config.authenticate().get("Authorization", "").replace("Bearer ", "")

        self.ai_client = OpenAI(
            api_key=self._token,
            base_url=f"{self._host}/serving-endpoints",
        )
        logger.info(f"ConverterService initialized with FMAPI at {self._host}")

    async def run_conversion(self, job: ConversionJob) -> ConversionJob:
        """Run conversion in-process.

        Args:
            job: ConversionJob to process

        Returns:
            Updated ConversionJob with results
        """
        job_id = job.job_id
        logger.info(f"Starting in-process conversion for {job_id}")

        try:
            # Stage 1: Parsing
            self.tracker.emit_status_change(job_id, "pending", "parsing")
            self.tracker.emit_progress(job_id, 10, "parsing", "Reading source file...")

            source_content = await self._read_source(job.source_path)
            self.tracker.emit_progress(job_id, 20, "parsing", f"Read {len(source_content)} bytes")

            # Pre-parse Informatica XML into structured context
            source_type_str = job.source_type.value if isinstance(job.source_type, SourceType) else job.source_type
            if source_type_str == "informatica_pc":
                self.tracker.emit_progress(job_id, 25, "parsing", "Parsing PowerCenter XML...")
                source_content = self._parse_informatica(source_content, job.source_path)
                self.tracker.emit_progress(job_id, 28, "parsing", "PowerCenter XML parsed into structured context")

            # Stage 2: Converting with AI
            self.tracker.emit_status_change(job_id, "parsing", "converting")
            self.tracker.emit_progress(job_id, 30, "converting", f"Calling {job.ai_model}...")

            conversion_result = await self._call_ai(
                source_content,
                job.source_type,
                job.ai_model,
                job.output_format,
            )
            self.tracker.emit_progress(job_id, 60, "converting", "AI conversion complete")

            # Stage 3: Agentic validation gate
            self.tracker.emit_status_change(job_id, "converting", "validating")
            self.tracker.emit_progress(job_id, 65, "validating", "Running agentic gate checks...")

            source_type_str = job.source_type.value if isinstance(job.source_type, SourceType) else job.source_type
            output_format_str = job.output_format.value if isinstance(job.output_format, OutputFormat) else (job.output_format or "pyspark")
            system_prompt = self.prompt_builder.build_system_prompt(source_type_str, output_format=output_format_str)
            user_prompt = self.prompt_builder.build_user_prompt(source_content, source_type_str)

            gate = AgenticGate(
                ai_client=self.ai_client,
                model_name=job.ai_model.value if isinstance(job.ai_model, AIModel) else job.ai_model,
                warehouse_id=self.settings.databricks_warehouse_id or None,
                databricks_host=self._host,
                databricks_token=self._token,
            )
            gate_report = await gate.validate_and_fix(
                conversion_result=conversion_result,
                source_content=source_content,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                source_type=source_type_str,
            )

            # Use the gate's (possibly corrected) result
            conversion_result = gate_report.final_result
            gate_info = gate_report.to_dict()

            if gate_report.passed:
                self.tracker.emit_progress(
                    job_id, 75, "validating",
                    f"Gate passed after {gate_report.total_attempts} attempt(s)"
                )
            else:
                self.tracker.emit_progress(
                    job_id, 75, "validating",
                    f"Gate completed with warnings after {gate_report.total_attempts} attempt(s)"
                )

            # Stage 4: Writing outputs
            self.tracker.emit_progress(job_id, 80, "writing", "Writing converted files...")

            output_path = await self._write_outputs(job_id, conversion_result)

            # Write gate report alongside outputs
            gate_report_path = f"{output_path}/gate_report.json"
            await self.volume_manager.write_file(
                gate_report_path,
                json.dumps(gate_info, indent=2).encode("utf-8"),
            )

            # Stage 5: Final score
            self.tracker.emit_progress(job_id, 90, "validating", "Calculating quality score...")

            quality_score = conversion_result.get("quality_notes", {}).get("score", 0.75)
            # Reduce score if gate didn't pass
            if not gate_report.passed:
                quality_score = min(quality_score, 0.60)

            # Complete
            self.tracker.emit_status_change(job_id, "validating", "completed")
            self.tracker.emit_progress(job_id, 100, "completed", "Conversion complete!")

            # Update job
            job.status = ConversionStatus.COMPLETED
            job.output_path = output_path
            job.quality_score = quality_score
            job.completed_at = datetime.utcnow()

            logger.info(f"Conversion {job_id} completed with score {quality_score}")
            return job

        except Exception as e:
            logger.error(f"Conversion {job_id} failed: {e}")
            self.tracker.emit_error(job_id, str(e))
            self.tracker.emit_status_change(job_id, job.status.value, "failed")

            job.status = ConversionStatus.FAILED
            job.error_message = str(e)
            return job

    async def _read_source(self, source_path: str) -> str:
        """Read source file content."""
        try:
            content_bytes = await self.volume_manager.read_file(source_path)
            return content_bytes.decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to read source: {e}")
            raise ValueError(f"Could not read source file: {e}")

    def _parse_informatica(self, xml_content: str, source_path: str) -> str:
        """Pre-parse Informatica PowerCenter XML into structured context for LLM.

        Instead of passing raw XML (verbose and hard for LLMs to process),
        parse it into a clean structured representation using the PowerCenter parser.
        """
        import tempfile
        try:
            from src.informatica.powercenter_parser import PowerCenterParser
        except ImportError:
            logger.warning("PowerCenter parser not available, passing raw XML")
            return xml_content

        try:
            # Write XML to temp file for parser (parser expects a file path)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as tmp:
                tmp.write(xml_content)
                tmp_path = tmp.name

            parser = PowerCenterParser(tmp_path)
            structured = parser.build_structured_context()
            logger.info(f"Parsed Informatica XML: {len(xml_content)} chars → {len(structured)} chars structured")
            return structured
        except Exception as e:
            logger.warning(f"Failed to pre-parse Informatica XML: {e}. Passing raw XML.")
            return xml_content

    async def _call_ai(
        self,
        source_content: str,
        source_type: SourceType,
        model: AIModel,
        output_format: OutputFormat = OutputFormat.PYSPARK,
    ) -> dict:
        """Call FMAPI to convert the source using config-driven prompts."""
        source_type_str = source_type.value if isinstance(source_type, SourceType) else source_type
        output_format_str = output_format.value if isinstance(output_format, OutputFormat) else (output_format or "pyspark")
        system_prompt = self.prompt_builder.build_system_prompt(source_type_str, output_format=output_format_str)
        user_prompt = self.prompt_builder.build_user_prompt(source_content, source_type_str)

        def normalize_content(content):
            """Normalize content to string - handles different model response formats.

            - Claude/OpenAI models: content is a string
            - Some OSS models: content may be a list of content blocks
            - Databricks GPT-OSS-120b: may return list format with text blocks
            - Databricks GPT-OSS-20b: returns reasoning format with summary
              [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "..."}]}]

            Always returns a string.
            """
            if content is None:
                return ""
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Handle list of content blocks
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type", "")

                        # Handle GPT-OSS-20b reasoning format
                        if block_type == "reasoning" and "summary" in block:
                            summary = block["summary"]
                            if isinstance(summary, list):
                                for item in summary:
                                    if isinstance(item, dict) and "text" in item:
                                        text_parts.append(str(item["text"]))
                                    elif isinstance(item, str):
                                        text_parts.append(item)
                                    else:
                                        text_parts.append(str(item))
                            elif isinstance(summary, str):
                                text_parts.append(summary)
                            else:
                                text_parts.append(str(summary))
                        # Handle standard text block
                        elif "text" in block:
                            text_parts.append(str(block["text"]))
                        # Handle content nested in block
                        elif "content" in block:
                            nested = block["content"]
                            if isinstance(nested, str):
                                text_parts.append(nested)
                            elif isinstance(nested, list):
                                # Recurse for nested content arrays
                                text_parts.append(normalize_content(nested))
                            else:
                                text_parts.append(str(nested))
                        else:
                            # Last resort: stringify the block
                            text_parts.append(str(block))
                    elif isinstance(block, str):
                        text_parts.append(block)
                    else:
                        text_parts.append(str(block))
                result = "\n".join(text_parts)
                # Ensure we return a string
                return result if isinstance(result, str) else str(result)
            else:
                return str(content)

        # Run in thread to avoid blocking
        def call_api():
            response = self.ai_client.chat.completions.create(
                model=model.value if isinstance(model, AIModel) else model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=8000,
            )
            raw_content = response.choices[0].message.content
            return normalize_content(raw_content)

        try:
            result_text = await asyncio.to_thread(call_api)

            # Ensure result_text is a string (defensive check)
            if not isinstance(result_text, str):
                logger.warning(f"normalize_content returned non-string type: {type(result_text)}")
                result_text = str(result_text) if result_text is not None else ""

            logger.info(f"Got AI response: {len(result_text)} characters")

            # Parse JSON response
            # Handle markdown code blocks
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            return json.loads(result_text.strip())

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response: {e}")
            # Return a fallback structure
            return {
                "notebooks": [{
                    "name": "conversion_output.py",
                    "description": "AI conversion output (unparsed)",
                    "code": f"# AI Response (manual parsing needed):\n'''\n{result_text}\n'''"
                }],
                "quality_notes": {
                    "score": 0.5,
                    "warnings": ["AI response was not valid JSON - manual review required"],
                    "manual_review": ["Full response needs parsing"]
                }
            }

        except Exception as e:
            logger.error(f"FMAPI call failed: {e}")
            raise ValueError(f"AI conversion failed: {e}")

    async def _write_outputs(self, job_id: str, result: dict) -> str:
        """Write conversion outputs to UC Volume."""
        output_base = f"{self.settings.outputs_path}/{job_id}"

        # Write workflow if present
        if "workflow" in result:
            workflow_path = f"{output_base}/workflow.json"
            await self.volume_manager.write_file(
                workflow_path,
                json.dumps(result["workflow"], indent=2).encode("utf-8"),
            )
            logger.info(f"Wrote workflow to {workflow_path}")

        # Write notebooks
        notebooks = result.get("notebooks", [])
        for notebook in notebooks:
            nb_name = notebook.get("name", "notebook.py")
            nb_path = f"{output_base}/notebooks/{nb_name}"
            nb_code = notebook.get("code", "# Empty notebook")

            # Add header comment
            header = f'''# Databricks notebook source
# MAGIC %md
# MAGIC # {notebook.get("description", "Converted notebook")}
# MAGIC
# MAGIC Generated by Conversion Control Tower

'''
            full_code = header + nb_code

            await self.volume_manager.write_file(
                nb_path,
                full_code.encode("utf-8"),
            )
            logger.info(f"Wrote notebook to {nb_path}")

        # Write quality report
        if "quality_notes" in result:
            report_path = f"{output_base}/quality_report.json"
            await self.volume_manager.write_file(
                report_path,
                json.dumps(result["quality_notes"], indent=2).encode("utf-8"),
            )
            logger.info(f"Wrote quality report to {report_path}")

        return output_base
