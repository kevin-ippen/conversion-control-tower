# Conversion Control Tower

AI-powered Databricks App for migrating legacy ETL workloads (SSIS, SQL Server, Informatica PowerCenter) to Databricks — with automated validation and quality scoring.

## What It Does

1. **Upload** source code (SSIS `.dtsx`, stored procedures, SQL scripts, Informatica XML)
2. **Convert** to Databricks PySpark notebooks, DLT pipelines, or dbt models using Foundation Model APIs
3. **Validate** converted output against original source data (synthetic, federated tables, or Lakeflow Connect replicas)
4. **Score** conversion quality across multiple dimensions (syntax, patterns, optimization, completeness)
5. **Promote** validated conversions through dev → QA → prod environments

<img width="2816" height="1536" alt="conversion-forge" src="https://github.com/user-attachments/assets/d06da02e-9472-4d64-831c-a5583fd149be" />


## Architecture

```
conversion-control-tower/
├── app/                              # Databricks App (FastAPI + React)
│   ├── app.yaml                      # App config (env vars, resource bindings)
│   ├── start.py                      # Entry point
│   ├── requirements.txt              # Python dependencies
│   ├── backend/                      # FastAPI backend
│   │   └── app/
│   │       ├── main.py               # FastAPI application
│   │       ├── auth.py               # OBO token authentication
│   │       ├── config.py             # Settings from environment
│   │       ├── models/               # Pydantic models
│   │       ├── routers/              # API endpoints
│   │       └── services/             # Business logic (converter, validator, etc.)
│   ├── frontend/                     # React + Vite + Tailwind
│   │   └── src/
│   │       ├── pages/                # Dashboard, Conversions, ConversionDetail, etc.
│   │       └── components/           # DataCompare, CodeCompare, ScoreCard, etc.
│   └── databricks/
│       ├── ddl/                      # Setup SQL scripts
│       └── notebooks/                # Validation pipeline notebooks
│           ├── conversion_runner.py  # AI-powered code conversion
│           ├── original_simulator.py # Simulates expected output from source logic
│           ├── converted_runner.py   # Executes converted Databricks code
│           └── data_comparator.py    # Compares expected vs actual output
├── src/                              # Shared parsing & conversion logic
│   ├── ssis/                         # SSIS .dtsx parser
│   ├── informatica/                  # PowerCenter XML parser
│   ├── extraction/                   # Schema extraction
│   ├── conversion/                   # Notebook & workflow generators
│   ├── synthetic/                    # Test data generation
│   └── validation/                   # Data validation utilities
├── config/                           # Conversion config + SSIS component mappings
├── samples/                          # Sample SSIS package + test data
├── databricks.yml                    # Databricks Asset Bundle definition
└── tests/                            # Test suite
```

## Prerequisites

- **Databricks Workspace** with Unity Catalog enabled
- **Databricks CLI** v0.200+ configured and authenticated
- **SQL Warehouse** (Serverless or Pro)
- **Foundation Model API** access (Claude Sonnet/Haiku endpoints)
- **Node.js 18+** (for frontend builds)

## Deployment

### 1. Configure Authentication

```bash
# Authenticate to your workspace
databricks auth login --host https://your-workspace.azuredatabricks.net

# Verify
databricks current-user me
```

### 2. Deploy the Bundle

```bash
# Find your SQL Warehouse ID
databricks sql warehouses list

# Deploy to dev
databricks bundle deploy --target dev --var warehouse_id=<YOUR_WAREHOUSE_ID>

# Get deployed job IDs
databricks bundle summary --target dev
```

### 3. Configure App Job IDs

After deploying, update `app/app.yaml` with the job IDs from the bundle summary:

```yaml
- name: CONVERSION_JOB_ID
  value: "<conversion_runner_job_id>"
- name: ORIGINAL_SIMULATOR_JOB_ID
  value: "<original_simulator_job_id>"
- name: CONVERTED_RUNNER_JOB_ID
  value: "<converted_runner_job_id>"
- name: DATA_COMPARATOR_JOB_ID
  value: "<data_comparator_job_id>"
```

Also set the `sql_warehouse_id` in the resources section.

Then redeploy:

```bash
databricks bundle deploy --target dev --var warehouse_id=<YOUR_WAREHOUSE_ID>
```

### 4. Run Setup DDL

```bash
# Create catalog, schemas, volume, and tracking tables
databricks bundle run setup_tracking_tables --target dev
```

### 5. Build Frontend (if modifying UI)

```bash
cd app/frontend
npm install
npm run build
# Output goes to app/backend/static/
```

### 6. Access the App

```bash
databricks apps list | grep conversion-control-tower
```

## Validation Pipeline

The validation pipeline runs as 3 chained Databricks jobs:

1. **Original Simulator** — Reads source code, extracts schema via LLM, generates synthetic test data with Faker, simulates expected output
2. **Converted Runner** — Executes the AI-converted Databricks notebook against the same test data, captures actual output
3. **Data Comparator** — Compares expected vs actual output cell-by-cell, generates match rate, schema diff, and sample mismatches

When a real source table is configured (UC table or federated table), the simulator skips synthetic data generation and reads directly from the source.

## Source Types Supported

| Source | Format | Parser |
|--------|--------|--------|
| SSIS | `.dtsx` (XML) | `src/ssis/dtsx_parser.py` |
| SQL Scripts | `.sql` | Direct LLM conversion |
| Stored Procedures | `.sql` | Direct LLM conversion |
| Informatica PowerCenter | `.xml` | `src/informatica/powercenter_parser.py` |

## Output Formats

| Target | Description |
|--------|-------------|
| PySpark | Databricks notebook with PySpark transformations |
| DLT/SDP | Spark Declarative Pipeline (DLT) definitions |
| dbt | dbt model SQL files |

## Environment Targets

| Target | Catalog | Mode |
|--------|---------|------|
| `dev` | `dev_conversion_tracker` | development |
| `qa` | `qa_conversion_tracker` | development |
| `prod` | `prod_conversion_tracker` | production (service principal) |

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Configuration

See [config/conversion_config.yaml](config/conversion_config.yaml) for conversion settings and [config/ssis_component_mappings.yaml](config/ssis_component_mappings.yaml) for SSIS-to-Spark component mappings.

For detailed deployment instructions, see [app/DEPLOYMENT.md](app/DEPLOYMENT.md).
