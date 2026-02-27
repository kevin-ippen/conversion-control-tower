"""
Application configuration loaded from environment variables.
"""

import os
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment."""

    # Databricks workspace
    databricks_host: str = ""
    databricks_token: Optional[str] = None

    # Unity Catalog - Tracking tables
    catalog: str = "dev_conversion_tracker"
    schema_name: str = "conversion_tracker"

    # Unity Catalog - Source data (for validation)
    source_catalog: str = "dev_conversion_tracker"
    source_schema: str = "source_data"

    # SQL Warehouse
    databricks_warehouse_id: str = ""

    # AI Model endpoint
    ai_model_endpoint: str = "databricks-claude-sonnet-4"

    # Optional federation
    federated_catalog: Optional[str] = None

    # Volume paths
    volume_base_path: str = "/Volumes/{catalog}/conversion_tracker/files"

    # App settings
    app_name: str = "Conversion Control Tower"
    debug: bool = False

    class Config:
        env_file = ".env"
        extra = "ignore"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # app.yaml sets SCHEMA env var, but pydantic maps schema_name to SCHEMA_NAME.
        # Bridge the gap: if SCHEMA is set in env, use it for schema_name.
        schema_env = os.environ.get("SCHEMA")
        if schema_env:
            object.__setattr__(self, "schema_name", schema_env)

    @property
    def tracking_schema(self) -> str:
        """Full schema path for tracking tables."""
        return f"{self.catalog}.{self.schema_name}"

    @property
    def volume_path(self) -> str:
        """Resolved volume path."""
        return self.volume_base_path.format(catalog=self.catalog)

    @property
    def uploads_path(self) -> str:
        """Path for uploaded source files."""
        return f"{self.volume_path}/uploads"

    @property
    def outputs_path(self) -> str:
        """Path for converted outputs."""
        return f"{self.volume_path}/outputs"

    @property
    def validation_path(self) -> str:
        """Path for validation artifacts."""
        return f"{self.volume_path}/validation"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
