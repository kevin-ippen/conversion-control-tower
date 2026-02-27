"""
Service for managing files in Unity Catalog Volumes.

Uses multiple strategies for file access:
1. Direct HTTP to Files API (most reliable)
2. SDK files.upload for writes
3. SQL-based fallback via Statement Execution API for reads/listing
"""

import io
import logging
import os
from typing import List, Optional
from pathlib import Path

import httpx
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

from ..config import Settings

logger = logging.getLogger(__name__)


class FileInfo:
    """Information about a file in UC Volume."""

    def __init__(self, name: str, path: str, size: int, is_directory: bool, modified_at: Optional[str] = None):
        self.name = name
        self.path = path
        self.size = size
        self.is_directory = is_directory
        self.modified_at = modified_at

    def dict(self):
        return {
            "name": self.name,
            "path": self.path,
            "size": self.size,
            "is_directory": self.is_directory,
            "modified_at": self.modified_at,
        }


class VolumeManager:
    """Manages files in Unity Catalog Volumes.

    Uses direct HTTP requests to the Files API for reads/listing,
    with SQL-based fallback via Statement Execution API.
    SDK files.upload is used for writes (proven to work).
    """

    def __init__(self, client: WorkspaceClient, settings: Settings):
        self.user_client = client  # User's OBO client (used for all operations)
        self.settings = settings

        host = os.environ.get("DATABRICKS_HOST", "")
        if not host.startswith("https://"):
            host = f"https://{host}"
        self.host = host

        # SP client only as fallback for background/admin operations
        self.client = WorkspaceClient(host=host)

        logger.info(f"VolumeManager init - host: {host}")
        logger.info(f"VolumeManager init - user auth: {client.config.auth_type}, sp auth: {self.client.config.auth_type}")

    def _get_user_headers(self) -> dict:
        """Get auth headers from the user's OBO client."""
        try:
            return self.user_client.config.authenticate()
        except Exception as e:
            logger.warning(f"Failed to get user auth headers: {e}")
            return {}

    def _get_sp_headers(self) -> dict:
        """Get auth headers from the SP client."""
        try:
            return self.client.config.authenticate()
        except Exception as e:
            logger.warning(f"Failed to get SP auth headers: {e}")
            return {}

    async def write_file(self, path: str, content: bytes) -> None:
        """Write content to a file in UC Volume.

        Uses direct HTTP PUT to the Files API with the SP client's token.
        The SP has been granted USE CATALOG + USE SCHEMA on the target catalog.
        Uses direct HTTP to avoid SDK BytesIO/len() bug in error parser.
        """
        import asyncio

        logger.info(f"Writing file to {path} ({len(content)} bytes)")

        api_path = path.lstrip("/")
        parent = str(Path(path).parent).lstrip("/")

        def do_upload():
            headers = self._get_sp_headers()
            if not headers:
                raise Exception("Failed to get SP auth headers for upload")

            with httpx.Client(timeout=120.0) as http:
                # Create parent directory
                dir_url = f"{self.host}/api/2.0/fs/directories/{parent}"
                try:
                    resp = http.put(dir_url, headers=headers)
                    logger.debug(f"Directory create response: {resp.status_code}")
                except Exception as e:
                    logger.debug(f"Directory creation skipped: {e}")

                # Upload file via PUT to Files API
                upload_url = f"{self.host}/api/2.0/fs/files/{api_path}"
                headers_with_ct = {**headers, "Content-Type": "application/octet-stream"}
                resp = http.put(upload_url, headers=headers_with_ct, content=content)
                if resp.status_code >= 400:
                    raise Exception(
                        f"Upload failed ({resp.status_code}): {resp.text[:500]}"
                    )

        try:
            await asyncio.to_thread(do_upload)
            logger.info(f"Successfully wrote {len(content)} bytes to {path}")
        except Exception as e:
            logger.error(f"Failed to write file to {path}: {e}")
            raise

    async def read_file(self, path: str) -> bytes:
        """Read content from a file in UC Volume.

        Strategy: Direct HTTP GET to Files API, trying user then SP auth.
        """
        import asyncio

        logger.info(f"Reading file from {path}")

        # Strip leading slash for the API path
        api_path = path.lstrip("/")

        # Try direct HTTP to Files API with both auth contexts
        for label, headers_fn in [("user", self._get_user_headers), ("sp", self._get_sp_headers)]:
            try:
                headers = await asyncio.to_thread(headers_fn)
                if not headers:
                    continue

                url = f"{self.host}/api/2.0/fs/files/{api_path}"

                def do_request():
                    with httpx.Client(timeout=30.0) as http:
                        resp = http.get(url, headers=headers)
                        resp.raise_for_status()
                        return resp.content

                content = await asyncio.to_thread(do_request)
                logger.info(f"Read {len(content)} bytes from {path} via HTTP ({label})")
                return content

            except Exception as e:
                logger.warning(f"HTTP read failed for {path} with {label}: {type(e).__name__}: {e}")
                continue

        raise Exception(f"All methods failed to read file from {path}")

    async def list_files(self, path: str) -> List[FileInfo]:
        """List files in a UC Volume directory.

        Strategy: Direct HTTP GET to Directories API, then SQL LIST fallback.
        """
        import asyncio

        logger.info(f"Listing files in {path}")

        # Strip leading slash for the API path
        api_path = path.rstrip("/").lstrip("/")

        # Try direct HTTP to Directories API
        for label, headers_fn in [("user", self._get_user_headers), ("sp", self._get_sp_headers)]:
            try:
                headers = await asyncio.to_thread(headers_fn)
                if not headers:
                    continue

                url = f"{self.host}/api/2.0/fs/directories/{api_path}"

                def do_request():
                    with httpx.Client(timeout=30.0) as http:
                        resp = http.get(url, headers=headers)
                        resp.raise_for_status()
                        return resp.json()

                data = await asyncio.to_thread(do_request)
                files = []
                for item in data.get("contents", []):
                    files.append(FileInfo(
                        name=item.get("name", ""),
                        path=item.get("path", ""),
                        size=item.get("file_size", 0) or 0,
                        is_directory=item.get("is_directory", False),
                        modified_at=str(item.get("modification_time")) if item.get("modification_time") else None,
                    ))
                logger.info(f"Listed {len(files)} files in {path} via HTTP ({label})")
                return files

            except Exception as e:
                logger.warning(f"HTTP list failed for {path} with {label}: {type(e).__name__}: {e}")
                continue

        # Fallback: SQL LIST via Statement Execution API
        logger.info(f"Falling back to SQL LIST for {path}")
        try:
            return await self._list_via_sql(path)
        except Exception as e:
            logger.error(f"SQL LIST fallback also failed for {path}: {e}")
            return []

    async def _list_via_sql(self, path: str) -> List[FileInfo]:
        """List directory contents using SQL LIST statement."""
        import asyncio

        warehouse_id = self.settings.databricks_warehouse_id
        if not warehouse_id:
            raise Exception("No warehouse_id configured for SQL fallback")

        query = f"LIST '{path}/'"

        def run_query():
            statement = self.user_client.statement_execution.execute_statement(
                warehouse_id=warehouse_id,
                statement=query,
                wait_timeout="30s",
            )

            while statement.status.state in (StatementState.PENDING, StatementState.RUNNING):
                import time
                time.sleep(0.5)
                statement = self.user_client.statement_execution.get_statement(
                    statement.statement_id
                )

            if statement.status.state == StatementState.FAILED:
                error = statement.status.error
                raise Exception(f"SQL LIST failed: {error.message if error else 'Unknown'}")

            if not statement.result or not statement.result.data_array:
                return []

            columns = [c.name for c in statement.manifest.schema.columns]
            files = []
            for row in statement.result.data_array:
                row_dict = dict(zip(columns, row))
                file_path = row_dict.get("path", "")
                file_name = Path(file_path).name
                size = int(row_dict.get("size", 0) or 0)
                is_dir = file_path.endswith("/")

                files.append(FileInfo(
                    name=file_name,
                    path=file_path.rstrip("/"),
                    size=size,
                    is_directory=is_dir,
                    modified_at=row_dict.get("modification_time"),
                ))
            return files

        return await asyncio.to_thread(run_query)

    async def delete_file(self, path: str) -> bool:
        """Delete a file from UC Volume via SP HTTP."""
        import asyncio

        logger.info(f"Deleting file {path}")
        try:
            api_path = path.lstrip("/")

            def do_delete():
                headers = self._get_sp_headers()
                with httpx.Client(timeout=30.0) as http:
                    resp = http.delete(f"{self.host}/api/2.0/fs/files/{api_path}", headers=headers)
                    resp.raise_for_status()

            await asyncio.to_thread(do_delete)
            logger.info(f"Successfully deleted {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {path}: {e}")
            return False

    async def create_directory(self, path: str) -> bool:
        """Create a directory in UC Volume via SP HTTP."""
        import asyncio

        logger.info(f"Creating directory {path}")
        try:
            api_path = path.lstrip("/")

            def do_mkdir():
                headers = self._get_sp_headers()
                with httpx.Client(timeout=30.0) as http:
                    resp = http.put(f"{self.host}/api/2.0/fs/directories/{api_path}", headers=headers)
                    resp.raise_for_status()

            await asyncio.to_thread(do_mkdir)
            logger.info(f"Successfully created directory {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            return False
