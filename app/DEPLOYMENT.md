# Conversion Control Tower - Deployment Guide

## Prerequisites

1. **Databricks Workspace** with Unity Catalog enabled
2. **Catalog Admin** privileges (for initial setup)
3. **SQL Warehouse** - Serverless or Pro SQL Warehouse
4. **Databricks CLI** configured with authentication

```bash
# Verify CLI is configured
databricks auth profiles
databricks current-user me
```

## Deployment Steps

### 1. Set Environment Variables

```bash
# Required
export DATABRICKS_HOST="https://your-workspace.azuredatabricks.net"
export DATABRICKS_WAREHOUSE_ID="your-warehouse-id"

# Optional - defaults to dev_conversion_tracker
export CATALOG="dev_conversion_tracker"
```

### 2. Run Initial Setup DDL

Execute the setup scripts in order using the SQL Warehouse:

```bash
# 1. Create catalog, schemas, and volume (requires catalog admin)
databricks sql query --warehouse-id $DATABRICKS_WAREHOUSE_ID \
  --file app/databricks/ddl/01_setup_catalog_and_volume.sql \
  --param catalog=$CATALOG

# 2. Create tracking tables
databricks sql query --warehouse-id $DATABRICKS_WAREHOUSE_ID \
  --file app/databricks/ddl/02_create_tracking_tables.sql \
  --param catalog=$CATALOG
```

Or run manually in the SQL Editor:
1. Open Databricks SQL Editor
2. Run `01_setup_catalog_and_volume.sql` (replace `${catalog}` with your catalog name)
3. Run `02_create_tracking_tables.sql`

### 3. Deploy with Databricks Asset Bundles

```bash
cd conversion-control-tower

# Validate bundle configuration
databricks bundle validate --target dev

# Deploy to development
databricks bundle deploy --target dev --var warehouse_id=$DATABRICKS_WAREHOUSE_ID

# Check deployment status
databricks bundle summary --target dev
```

### 4. Start the App

```bash
# Get app URL
databricks apps list | grep conversion-control-tower

# Or start from CLI
databricks apps start conversion-control-tower
```

## Target Environments

| Target | Catalog | Mode |
|--------|---------|------|
| dev | dev_conversion_tracker | development |
| qa | qa_conversion_tracker | development |
| prod | prod_conversion_tracker | production |

Deploy to different environments:

```bash
# QA deployment
databricks bundle deploy --target qa --var warehouse_id=$DATABRICKS_WAREHOUSE_ID

# Production deployment (requires service principal)
databricks bundle deploy --target prod --var warehouse_id=$DATABRICKS_WAREHOUSE_ID
```

## Verification

### Check Resources Created

```sql
-- Verify catalog and schemas
SHOW SCHEMAS IN ${catalog};

-- Verify tracking tables
SHOW TABLES IN ${catalog}.conversion_tracker;

-- Verify volume
SHOW VOLUMES IN ${catalog}.conversion_tracker;

-- Check table access
SELECT * FROM ${catalog}.conversion_tracker.conversion_jobs LIMIT 5;
```

### Test App Health

```bash
# Get app URL
APP_URL=$(databricks apps get conversion-control-tower --output json | jq -r '.url')

# Test health endpoint
curl -s "$APP_URL/health" | jq
```

### Test API Endpoints

```bash
# List conversions (should return empty array initially)
curl -s "$APP_URL/api/conversions" \
  -H "Authorization: Bearer $(databricks auth token)" | jq

# Check analytics overview
curl -s "$APP_URL/api/analytics/overview" \
  -H "Authorization: Bearer $(databricks auth token)" | jq
```

## Permissions Summary

### App Permissions (app.yaml)

| Permission | Purpose |
|------------|---------|
| sql | Execute SQL queries via warehouse |
| files | Read/write UC Volumes |
| serving | Call AI model endpoints |
| jobs | Create and manage Databricks Jobs |

### Unity Catalog Grants

The DDL scripts grant these permissions to `account users`:

| Object | Grants |
|--------|--------|
| Catalog | USAGE, CREATE SCHEMA |
| Schemas | USAGE, CREATE TABLE, CREATE VIEW |
| Volume | READ VOLUME, WRITE VOLUME |
| Tables | SELECT, MODIFY |
| Views | SELECT |

### Production Considerations

For production deployments:

1. **Restrict grants** to specific groups instead of `account users`
2. **Use service principal** for app execution (configured in `databricks.yml`)
3. **Enable audit logging** via system tables
4. **Set up alerting** for failed conversion jobs

## Troubleshooting

### Common Issues

**"Catalog not found"**
- Ensure you have CREATE CATALOG privileges
- Run `01_setup_catalog_and_volume.sql` first

**"Permission denied on volume"**
- Check volume grants: `SHOW GRANTS ON VOLUME ${catalog}.conversion_tracker.files`
- Grant access: `GRANT READ VOLUME, WRITE VOLUME ON VOLUME ... TO user`

**"SQL Warehouse not found"**
- Verify warehouse ID: `databricks sql warehouses list`
- Ensure warehouse is running

**"App fails to start"**
- Check app logs: `databricks apps logs conversion-control-tower`
- Verify all required resources exist

### Logs and Debugging

```bash
# View app logs
databricks apps logs conversion-control-tower --follow

# Check job run status
databricks jobs runs list --job-id <conversion_runner_job_id>

# Query status events
SELECT * FROM ${catalog}.conversion_tracker.status_events
ORDER BY created_at DESC
LIMIT 20;
```

## File Structure Reference

```
app/
├── app.yaml                    # Databricks App configuration
├── start.py                    # Entry point
├── requirements.txt            # Python dependencies
├── backend/
│   └── app/
│       ├── main.py             # FastAPI application
│       ├── config.py           # Settings from environment
│       ├── auth.py             # OBO token handling
│       ├── models/             # Pydantic models
│       ├── routers/            # API endpoints
│       └── services/           # Business logic
├── frontend/
│   └── src/                    # React application
└── databricks/
    ├── ddl/                    # Setup SQL scripts
    └── notebooks/              # Conversion runner notebook
```

## Next Steps

After deployment:

1. **Upload test SSIS package** via the app UI
2. **Run a test conversion** to verify end-to-end flow
3. **Check quality scores** and validation results
4. **Test promotion workflow** from Dev to QA
