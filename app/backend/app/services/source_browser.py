"""
Service for browsing source files in Workspace, UC Volumes, and Repos.

Provides a unified interface for listing files across different Databricks locations.
"""

import asyncio
import logging
from typing import List, Optional
from dataclasses import dataclass

import httpx
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)

# File extensions relevant for conversion
CONVERTIBLE_EXTENSIONS = {'.dtsx', '.sql', '.xml', '.txt', '.py'}


@dataclass
class BrowseItem:
    """A file or directory entry from browsing."""
    name: str
    path: str
    is_directory: bool
    size: int = 0
    modified_at: Optional[str] = None

    def dict(self):
        return {
            "name": self.name,
            "path": self.path,
            "is_directory": self.is_directory,
            "size": self.size,
            "modified_at": self.modified_at,
        }


class SourceBrowser:
    """Browse files across Workspace, UC Volumes, and Repos."""

    def __init__(self, client: WorkspaceClient):
        self.client = client
        self.host = str(client.config.host or "").rstrip("/")

    def _get_headers(self) -> dict:
        try:
            return self.client.config.authenticate()
        except Exception as e:
            logger.warning(f"Failed to get auth headers: {e}")
            return {}

    async def list_workspace(self, path: str) -> List[BrowseItem]:
        """List files and folders in a Workspace path via Workspace API."""
        logger.info(f"Browsing workspace path: {path}")

        def do_list():
            headers = self._get_headers()
            if not headers:
                return []
            url = f"{self.host}/api/2.0/workspace/list"
            with httpx.Client(timeout=15.0) as http:
                resp = http.get(url, headers=headers, params={"path": path})
                resp.raise_for_status()
                return resp.json().get("objects", [])

        try:
            objects = await asyncio.to_thread(do_list)
            items = []
            for obj in objects:
                name = obj.get("path", "").rsplit("/", 1)[-1]
                is_dir = obj.get("object_type") in ("DIRECTORY", "REPO")
                ext = f".{name.rsplit('.', 1)[-1].lower()}" if '.' in name else ''
                # Show directories and convertible files
                if is_dir or ext in CONVERTIBLE_EXTENSIONS:
                    items.append(BrowseItem(
                        name=name,
                        path=obj.get("path", ""),
                        is_directory=is_dir,
                        size=obj.get("size", 0) or 0,
                        modified_at=str(obj.get("modified_at")) if obj.get("modified_at") else None,
                    ))
            return sorted(items, key=lambda x: (not x.is_directory, x.name.lower()))
        except Exception as e:
            logger.error(f"Failed to browse workspace {path}: {e}")
            return []

    async def list_volume(self, path: str) -> List[BrowseItem]:
        """List files and folders in a UC Volume path via Files API."""
        logger.info(f"Browsing volume path: {path}")

        api_path = path.rstrip("/").lstrip("/")

        def do_list():
            headers = self._get_headers()
            if not headers:
                return []
            url = f"{self.host}/api/2.0/fs/directories/{api_path}"
            with httpx.Client(timeout=15.0) as http:
                resp = http.get(url, headers=headers)
                resp.raise_for_status()
                return resp.json().get("contents", [])

        try:
            contents = await asyncio.to_thread(do_list)
            items = []
            for item in contents:
                name = item.get("name", "")
                is_dir = item.get("is_directory", False)
                ext = f".{name.rsplit('.', 1)[-1].lower()}" if '.' in name else ''
                if is_dir or ext in CONVERTIBLE_EXTENSIONS:
                    items.append(BrowseItem(
                        name=name,
                        path=item.get("path", ""),
                        is_directory=is_dir,
                        size=item.get("file_size", 0) or 0,
                        modified_at=str(item.get("modification_time")) if item.get("modification_time") else None,
                    ))
            return sorted(items, key=lambda x: (not x.is_directory, x.name.lower()))
        except Exception as e:
            logger.error(f"Failed to browse volume {path}: {e}")
            return []

    async def list_repo(self, repo_path: str, sub_path: str = "") -> List[BrowseItem]:
        """List files in a Databricks Repo path.

        Repos appear under /Repos/<user>/<repo-name>/ in the Workspace namespace,
        so we reuse the workspace listing API.
        """
        full_path = f"{repo_path.rstrip('/')}/{sub_path}".rstrip("/")
        return await self.list_workspace(full_path)

    async def read_file_preview(self, path: str, max_bytes: int = 2000) -> str:
        """Read the first N bytes of a file for format detection.

        Tries Workspace API for workspace files, Files API for volume files.
        """
        # Determine which API to use based on path
        if path.startswith("/Volumes"):
            return await self._read_volume_preview(path, max_bytes)
        else:
            return await self._read_workspace_preview(path, max_bytes)

    async def _read_workspace_preview(self, path: str, max_bytes: int) -> str:
        """Read preview from a Workspace file."""
        def do_read():
            headers = self._get_headers()
            if not headers:
                return ""
            url = f"{self.host}/api/2.0/workspace/export"
            with httpx.Client(timeout=15.0) as http:
                resp = http.get(url, headers=headers, params={
                    "path": path,
                    "format": "SOURCE",
                    "direct_download": True,
                })
                resp.raise_for_status()
                return resp.text[:max_bytes]

        try:
            return await asyncio.to_thread(do_read)
        except Exception as e:
            logger.warning(f"Failed to read workspace preview {path}: {e}")
            return ""

    async def _read_volume_preview(self, path: str, max_bytes: int) -> str:
        """Read preview from a UC Volume file."""
        api_path = path.lstrip("/")

        def do_read():
            headers = self._get_headers()
            if not headers:
                return ""
            url = f"{self.host}/api/2.0/fs/files/{api_path}"
            with httpx.Client(timeout=15.0) as http:
                # Use Range header to limit download
                req_headers = {**headers, "Range": f"bytes=0-{max_bytes}"}
                resp = http.get(url, headers=req_headers)
                resp.raise_for_status()
                return resp.text[:max_bytes]

        try:
            return await asyncio.to_thread(do_read)
        except Exception as e:
            logger.warning(f"Failed to read volume preview {path}: {e}")
            return ""
