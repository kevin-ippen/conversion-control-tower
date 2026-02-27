-- Conversion Control Tower - Unity Catalog Tracking Tables
-- Run this script to create the required schema and tables

-- Create catalog (if needed)
-- CREATE CATALOG IF NOT EXISTS dev_conversion_tracker;

-- Use the catalog
USE CATALOG ${catalog};

-- Create schema for tracking tables
CREATE SCHEMA IF NOT EXISTS conversion_tracker
COMMENT 'Schema for SQL Server to Databricks conversion tracking';

USE SCHEMA conversion_tracker;

-- =============================================================================
-- CONVERSION JOBS
-- =============================================================================

CREATE TABLE IF NOT EXISTS conversion_jobs (
    job_id STRING NOT NULL COMMENT 'Unique job identifier (UUID)',
    job_name STRING NOT NULL COMMENT 'Human-readable job name',
    source_type STRING NOT NULL COMMENT 'Type of source: ssis, stored_proc, sql_script',
    source_path STRING NOT NULL COMMENT 'UC Volume path to source file',
    output_path STRING COMMENT 'UC Volume path to converted output',
    status STRING NOT NULL DEFAULT 'pending' COMMENT 'Job status: pending, parsing, converting, validating, completed, failed, cancelled',
    quality_score DOUBLE COMMENT 'Overall quality score (0.0 to 1.0)',
    created_by STRING COMMENT 'User who created the job',
    created_at TIMESTAMP NOT NULL DEFAULT current_timestamp() COMMENT 'Job creation timestamp',
    updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp() COMMENT 'Last update timestamp',
    completed_at TIMESTAMP COMMENT 'Job completion timestamp',
    error_message STRING COMMENT 'Error message if job failed',
    metadata MAP<STRING, STRING> COMMENT 'Additional job metadata',
    databricks_run_id STRING COMMENT 'Associated Databricks Jobs run ID',
    CONSTRAINT pk_conversion_jobs PRIMARY KEY (job_id)
)
USING DELTA
COMMENT 'Tracks all conversion jobs submitted to the system'
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true'
);

-- =============================================================================
-- CONVERSION FILES
-- =============================================================================

CREATE TABLE IF NOT EXISTS conversion_files (
    file_id STRING NOT NULL COMMENT 'Unique file identifier (UUID)',
    job_id STRING NOT NULL COMMENT 'Parent job ID',
    source_file STRING NOT NULL COMMENT 'Source file path',
    target_file STRING COMMENT 'Generated output file path',
    file_type STRING NOT NULL COMMENT 'Output type: notebook, workflow, sql, dlt',
    status STRING NOT NULL DEFAULT 'pending' COMMENT 'File conversion status',
    quality_score DOUBLE COMMENT 'File-level quality score',
    validation_details STRING COMMENT 'JSON blob with validation details',
    created_at TIMESTAMP NOT NULL DEFAULT current_timestamp(),
    updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp(),
    CONSTRAINT pk_conversion_files PRIMARY KEY (file_id),
    CONSTRAINT fk_files_job FOREIGN KEY (job_id) REFERENCES conversion_jobs(job_id)
)
USING DELTA
COMMENT 'Tracks individual files within conversion jobs';

-- =============================================================================
-- VALIDATION RESULTS
-- =============================================================================

CREATE TABLE IF NOT EXISTS validation_results (
    validation_id STRING NOT NULL COMMENT 'Unique validation ID (UUID)',
    job_id STRING NOT NULL COMMENT 'Associated job ID',
    check_name STRING NOT NULL COMMENT 'Name of the validation check',
    passed BOOLEAN NOT NULL COMMENT 'Whether the check passed',
    expected STRING COMMENT 'Expected value',
    actual STRING COMMENT 'Actual value',
    message STRING COMMENT 'Human-readable result message',
    severity STRING NOT NULL DEFAULT 'error' COMMENT 'Severity: error, warning, info',
    category STRING NOT NULL COMMENT 'Category: completeness, accuracy, logic, error_handling',
    created_at TIMESTAMP NOT NULL DEFAULT current_timestamp(),
    CONSTRAINT pk_validation_results PRIMARY KEY (validation_id),
    CONSTRAINT fk_validation_job FOREIGN KEY (job_id) REFERENCES conversion_jobs(job_id)
)
USING DELTA
COMMENT 'Stores validation check results for each job';

-- =============================================================================
-- PROMOTION HISTORY
-- =============================================================================

CREATE TABLE IF NOT EXISTS promotion_history (
    promotion_id STRING NOT NULL COMMENT 'Unique promotion ID (UUID)',
    job_id STRING NOT NULL COMMENT 'Associated job ID',
    from_environment STRING NOT NULL COMMENT 'Source environment: dev, qa, prod',
    to_environment STRING NOT NULL COMMENT 'Target environment: dev, qa, prod',
    promoted_by STRING NOT NULL COMMENT 'User who initiated promotion',
    promoted_at TIMESTAMP NOT NULL DEFAULT current_timestamp() COMMENT 'Promotion request timestamp',
    approval_status STRING NOT NULL DEFAULT 'pending' COMMENT 'Status: pending, approved, rejected',
    approver STRING COMMENT 'User who approved/rejected',
    approved_at TIMESTAMP COMMENT 'Approval timestamp',
    rejection_reason STRING COMMENT 'Reason for rejection',
    notes STRING COMMENT 'Additional notes',
    deployed_job_id STRING COMMENT 'Databricks Job ID in target environment',
    deployed_job_url STRING COMMENT 'URL to deployed job',
    CONSTRAINT pk_promotion_history PRIMARY KEY (promotion_id),
    CONSTRAINT fk_promotion_job FOREIGN KEY (job_id) REFERENCES conversion_jobs(job_id)
)
USING DELTA
COMMENT 'Tracks promotion requests and approvals through dev/qa/prod';

-- =============================================================================
-- APPROVAL CONFIGURATION
-- =============================================================================

CREATE TABLE IF NOT EXISTS approval_config (
    environment STRING NOT NULL COMMENT 'Environment: qa, prod',
    required_approvers ARRAY<STRING> COMMENT 'List of users who can approve',
    min_quality_score DOUBLE DEFAULT 0.70 COMMENT 'Minimum quality score to promote',
    auto_approve BOOLEAN DEFAULT FALSE COMMENT 'Whether to auto-approve if score meets threshold',
    updated_at TIMESTAMP NOT NULL DEFAULT current_timestamp(),
    updated_by STRING COMMENT 'Last user to update config',
    CONSTRAINT pk_approval_config PRIMARY KEY (environment)
)
USING DELTA
COMMENT 'Configuration for promotion approval workflow';

-- Insert default approval config
MERGE INTO approval_config AS target
USING (
    SELECT 'qa' AS environment, 0.70 AS min_quality_score, FALSE AS auto_approve
    UNION ALL
    SELECT 'prod' AS environment, 0.80 AS min_quality_score, FALSE AS auto_approve
) AS source
ON target.environment = source.environment
WHEN NOT MATCHED THEN
    INSERT (environment, min_quality_score, auto_approve, updated_at)
    VALUES (source.environment, source.min_quality_score, source.auto_approve, current_timestamp());

-- =============================================================================
-- STATUS EVENTS (for real-time SSE streaming)
-- =============================================================================

CREATE TABLE IF NOT EXISTS status_events (
    event_id STRING NOT NULL COMMENT 'Unique event ID (UUID)',
    job_id STRING NOT NULL COMMENT 'Associated job ID',
    event_type STRING NOT NULL COMMENT 'Event type: status_change, progress, log, error',
    event_data STRING NOT NULL COMMENT 'JSON blob with event details',
    created_at TIMESTAMP NOT NULL DEFAULT current_timestamp(),
    CONSTRAINT pk_status_events PRIMARY KEY (event_id),
    CONSTRAINT fk_events_job FOREIGN KEY (job_id) REFERENCES conversion_jobs(job_id)
)
USING DELTA
COMMENT 'Real-time status events for job monitoring'
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true'
);

-- =============================================================================
-- VIEWS
-- =============================================================================

-- View: Active jobs (not completed or failed)
CREATE OR REPLACE VIEW active_jobs AS
SELECT *
FROM conversion_jobs
WHERE status NOT IN ('completed', 'failed', 'cancelled');

-- View: Jobs pending approval
CREATE OR REPLACE VIEW pending_approvals AS
SELECT
    p.*,
    j.job_name,
    j.quality_score,
    j.source_type
FROM promotion_history p
JOIN conversion_jobs j ON p.job_id = j.job_id
WHERE p.approval_status = 'pending';

-- View: Quality score summary by category
CREATE OR REPLACE VIEW quality_summary AS
SELECT
    j.job_id,
    j.job_name,
    j.quality_score AS overall_score,
    AVG(CASE WHEN v.category = 'completeness' AND v.passed THEN 1.0 ELSE 0.0 END) AS completeness_score,
    AVG(CASE WHEN v.category = 'accuracy' AND v.passed THEN 1.0 ELSE 0.0 END) AS accuracy_score,
    AVG(CASE WHEN v.category = 'logic' AND v.passed THEN 1.0 ELSE 0.0 END) AS logic_score,
    AVG(CASE WHEN v.category = 'error_handling' AND v.passed THEN 1.0 ELSE 0.0 END) AS error_handling_score
FROM conversion_jobs j
LEFT JOIN validation_results v ON j.job_id = v.job_id
GROUP BY j.job_id, j.job_name, j.quality_score;

-- =============================================================================
-- INDEXES (for performance)
-- =============================================================================

-- Note: Delta Lake handles indexes automatically via ZORDER
-- Run OPTIMIZE with ZORDER periodically for large tables

-- OPTIMIZE conversion_jobs ZORDER BY (status, created_at);
-- OPTIMIZE validation_results ZORDER BY (job_id, category);
-- OPTIMIZE status_events ZORDER BY (job_id, created_at);

-- =============================================================================
-- GRANTS
-- =============================================================================

-- Grant schema usage to all users in account
GRANT USAGE ON SCHEMA ${catalog}.conversion_tracker TO `account users`;

-- Grant table access for app users
GRANT SELECT, MODIFY ON TABLE ${catalog}.conversion_tracker.conversion_jobs TO `account users`;
GRANT SELECT, MODIFY ON TABLE ${catalog}.conversion_tracker.conversion_files TO `account users`;
GRANT SELECT, MODIFY ON TABLE ${catalog}.conversion_tracker.validation_results TO `account users`;
GRANT SELECT, MODIFY ON TABLE ${catalog}.conversion_tracker.promotion_history TO `account users`;
GRANT SELECT, MODIFY ON TABLE ${catalog}.conversion_tracker.approval_config TO `account users`;
GRANT SELECT, MODIFY ON TABLE ${catalog}.conversion_tracker.status_events TO `account users`;

-- Grant view access
GRANT SELECT ON VIEW ${catalog}.conversion_tracker.active_jobs TO `account users`;
GRANT SELECT ON VIEW ${catalog}.conversion_tracker.pending_approvals TO `account users`;
GRANT SELECT ON VIEW ${catalog}.conversion_tracker.quality_summary TO `account users`;
