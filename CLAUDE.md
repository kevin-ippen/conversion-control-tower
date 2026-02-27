# Conversion Control Tower

## Project Structure
- `app/` - Databricks App (FastAPI backend + React frontend)
- `app/frontend/` - React + Vite + Tailwind frontend
- `app/backend/` - FastAPI backend with conversion services
- `app/databricks/` - Notebooks and DDL for Databricks jobs
- `src/` - Shared conversion logic (prompt engine, config loader, parsers)
- `config/` - Conversion config YAML + examples
- `databricks.yml` - DABs bundle definition

## Deployment
- Frontend builds to `app/backend/static/` via `cd app/frontend && npm run build`
- Deploy with `databricks bundle deploy --target dev --var warehouse_id=<ID>`
- App name: `conversion-control-tower`
- After deploy, update `app/app.yaml` job IDs from `databricks bundle summary`

## Key Services
- `converter.py` - AI-powered conversion using Foundation Model APIs
- `agentic_gate.py` - Post-conversion validation with EXPLAIN + auto-retry
- `code_quality_analyzer.py` - Multi-dimensional code quality scoring
- `prompt_builder.py` - Config-driven prompt construction
- `validator.py` - Orchestrates quality validation
- `job_runner.py` - Manages Databricks Job runs for validation pipeline

## Validation Pipeline (3 jobs)
1. `original_simulator.py` - Extracts schema from source, generates synthetic data, simulates expected output
2. `converted_runner.py` - Runs converted code against test data
3. `data_comparator.py` - Compares expected vs actual output

## Important Notes
- All notebooks use pandas + pyarrow for I/O (not Spark DataFrames) to avoid hangs on single-node clusters
- `dict.get("key", default)` returns None when key exists with null value — use `dict.get("key") or default`
- The app uses OBO (On-Behalf-Of) authentication — users access resources with their own permissions
