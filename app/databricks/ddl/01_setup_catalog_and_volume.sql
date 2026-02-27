-- Conversion Control Tower - Initial Setup
-- Run this FIRST to create the catalog, schemas, and volumes
-- Requires: CATALOG admin privileges

-- =============================================================================
-- CATALOG SETUP
-- =============================================================================

-- Create catalog for conversion tracking (one per environment)
CREATE CATALOG IF NOT EXISTS ${catalog}
COMMENT 'Catalog for SQL Server to Databricks conversion tracking';

-- Grant catalog usage
GRANT USAGE ON CATALOG ${catalog} TO `account users`;
GRANT CREATE SCHEMA ON CATALOG ${catalog} TO `account users`;

USE CATALOG ${catalog};

-- =============================================================================
-- SCHEMA SETUP
-- =============================================================================

-- Schema for tracking tables
CREATE SCHEMA IF NOT EXISTS conversion_tracker
COMMENT 'Schema for conversion tracking tables and metadata';

-- Schema for source/test data
CREATE SCHEMA IF NOT EXISTS source_data
COMMENT 'Schema for source and test data used in validation';

-- Grant schema usage
GRANT USAGE, CREATE TABLE, CREATE VIEW, CREATE FUNCTION ON SCHEMA conversion_tracker TO `account users`;
GRANT USAGE, CREATE TABLE, CREATE VIEW ON SCHEMA source_data TO `account users`;

-- =============================================================================
-- VOLUME SETUP
-- =============================================================================

-- Create managed volume for file storage
CREATE VOLUME IF NOT EXISTS conversion_tracker.files
COMMENT 'Storage for uploaded source files and converted outputs';

-- Grant volume access
GRANT READ VOLUME, WRITE VOLUME ON VOLUME conversion_tracker.files TO `account users`;

-- =============================================================================
-- DIRECTORY STRUCTURE
-- =============================================================================

-- Note: Directories are created automatically when files are written
-- The expected structure is:
--
-- /Volumes/${catalog}/conversion_tracker/files/
-- ├── uploads/          # User-uploaded source files
-- │   └── {job_id}/
-- │       └── *.dtsx, *.sql
-- ├── outputs/          # Converted outputs
-- │   └── {job_id}/
-- │       ├── workflow.json
-- │       └── notebooks/
-- ├── validation/       # Validation artifacts
-- │   └── {job_id}/
-- │       └── validation_report.json
-- └── test_data/        # Shared test datasets
--     └── *.parquet

-- =============================================================================
-- VERIFICATION
-- =============================================================================

-- Verify setup
SELECT 'Catalog created' AS step, current_catalog() AS value
UNION ALL
SELECT 'Schemas', CONCAT_WS(', ', COLLECT_LIST(schema_name)) FROM system.information_schema.schemata WHERE catalog_name = current_catalog()
UNION ALL
SELECT 'Volume path', '/Volumes/' || current_catalog() || '/conversion_tracker/files';
