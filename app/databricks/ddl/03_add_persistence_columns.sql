-- Migration: Add columns for persistence and lineage tracking
-- Run this script after 02_create_tracking_tables.sql

USE CATALOG ${catalog};
USE SCHEMA conversion_tracker;

-- =============================================================================
-- Add new columns to conversion_jobs for full persistence
-- =============================================================================

-- AI model used for conversion
ALTER TABLE conversion_jobs ADD COLUMN IF NOT EXISTS
    ai_model STRING COMMENT 'AI model used: databricks-claude-haiku-4-5, databricks-gpt-oss-20b, etc';

-- Custom instructions passed to AI
ALTER TABLE conversion_jobs ADD COLUMN IF NOT EXISTS
    conversion_instructions STRING COMMENT 'Custom instructions passed to AI model';

-- Refinement lineage - parent job ID if this is a refinement
ALTER TABLE conversion_jobs ADD COLUMN IF NOT EXISTS
    refinement_of STRING COMMENT 'Parent job_id if this is a refinement job';

-- Code previews for quick UI rendering
ALTER TABLE conversion_jobs ADD COLUMN IF NOT EXISTS
    source_code_preview STRING COMMENT 'First 2000 chars of source for quick preview';

ALTER TABLE conversion_jobs ADD COLUMN IF NOT EXISTS
    output_code_preview STRING COMMENT 'First 2000 chars of output for quick preview';

-- Full quality report as JSON
ALTER TABLE conversion_jobs ADD COLUMN IF NOT EXISTS
    quality_report_json STRING COMMENT 'Full quality report as JSON string';

-- =============================================================================
-- Create data_comparisons table for validation results
-- =============================================================================

CREATE TABLE IF NOT EXISTS data_comparisons (
    comparison_id STRING NOT NULL COMMENT 'Unique comparison ID (UUID)',
    job_id STRING NOT NULL COMMENT 'Associated job ID',
    comparison_type STRING NOT NULL COMMENT 'Type: schema, row_count, data_values',
    table_name STRING NOT NULL COMMENT 'Table being compared',
    expected_value STRING COMMENT 'Expected value from original/synthetic',
    actual_value STRING COMMENT 'Actual value from converted code output',
    match_status STRING NOT NULL COMMENT 'Status: match, mismatch, missing',
    diff_details STRING COMMENT 'JSON with detailed differences',
    created_at TIMESTAMP DEFAULT current_timestamp(),
    CONSTRAINT pk_data_comparisons PRIMARY KEY (comparison_id)
)
USING DELTA
COMMENT 'Stores data comparison results for conversion validation'
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true'
);

-- =============================================================================
-- Create view for refinement lineage
-- =============================================================================

CREATE OR REPLACE VIEW refinement_lineage AS
WITH RECURSIVE lineage AS (
    -- Base case: original jobs (no refinement_of)
    SELECT
        job_id,
        job_name,
        status,
        quality_score,
        refinement_of,
        created_at,
        0 AS depth
    FROM conversion_jobs
    WHERE refinement_of IS NULL

    UNION ALL

    -- Recursive: refinements
    SELECT
        c.job_id,
        c.job_name,
        c.status,
        c.quality_score,
        c.refinement_of,
        c.created_at,
        l.depth + 1
    FROM conversion_jobs c
    JOIN lineage l ON c.refinement_of = l.job_id
)
SELECT * FROM lineage;

-- =============================================================================
-- Grant permissions for new table
-- =============================================================================

GRANT SELECT, MODIFY ON TABLE ${catalog}.conversion_tracker.data_comparisons TO `account users`;
GRANT SELECT ON VIEW ${catalog}.conversion_tracker.refinement_lineage TO `account users`;
