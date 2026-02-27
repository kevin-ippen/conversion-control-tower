# SSIS Package: SalesDataETL - Conversion Guide

## Overview
This is a medium-complexity SSIS package that performs a typical data warehouse ETL process for sales data. It demonstrates many common SSIS patterns that you'll need to convert to Databricks.

---

## Package Summary

| Component | Count | Databricks Equivalent |
|-----------|-------|----------------------|
| Connection Managers | 3 | Unity Catalog connections, Secrets |
| Variables | 8 | Widgets, notebook parameters, Spark configs |
| Execute SQL Tasks | 6 | `spark.sql()`, Delta Lake operations |
| Data Flow Tasks | 2 | Spark DataFrames, Delta Lake |
| Sequence Container | 1 | Workflow task groups |
| Event Handlers | 1 | Try/except blocks, workflow error handling |
| Send Mail Tasks | 2 | Notification tasks, webhooks |

---

## Control Flow Structure

```
1. Log Package Start (Execute SQL)
        ↓
2. Truncate Staging Tables (Execute SQL)
        ↓
3. Get Last Extract Date (Execute SQL → Variables)
        ↓
4. [SEQUENCE CONTAINER: Main ETL Process]
    ├── Data Flow: Extract Transform Sales
    │       ↓
    └── Data Flow: Customer SCD Type 2
        ↓
5. Update Watermark (Execute SQL with MERGE)
        ↓
6. Log Package Success (Execute SQL)
        ↓
7. Send Success Email

[OnError Event Handler]
    → Log Error → Update Batch Failed → Send Failure Email
```

---

## Data Flow 1: Extract Transform Sales

### Components to Convert:

#### 1. **OLE DB Source** → Spark DataFrame read
```python
# SSIS: Parameterized SQL with date range
# Databricks:
df_sales = spark.read.jdbc(
    url=jdbc_url,
    table="(SELECT ... FROM dbo.Sales WHERE SaleDate >= ? AND SaleDate < ?) AS t",
    properties={"user": user, "password": password}
)
```

#### 2. **Lookup Transforms** → DataFrame joins or broadcast joins
```python
# SSIS: Lookup against dim.Customer, dim.Product, dim.Date
# Databricks:
df_customer = spark.table("dim.customer").filter("IsCurrent = 1")
df_enriched = df_sales.join(
    broadcast(df_customer),  # Use broadcast for small dimensions
    on="CustomerID",
    how="left"
)
```

#### 3. **Derived Column** → withColumn / select with expressions
```python
# SSIS expressions → Spark SQL expressions
df = df.withColumn("ProfitAmount", col("NetAmount") - (col("Quantity") * col("StandardCost")))
df = df.withColumn("ProfitMarginPct", 
    when(col("NetAmount") == 0, 0)
    .otherwise(((col("NetAmount") - (col("Quantity") * col("StandardCost"))) / col("NetAmount")) * 100)
)
df = df.withColumn("SaleCategory",
    when(col("NetAmount") < 100, "Small")
    .when(col("NetAmount") < 1000, "Medium")
    .when(col("NetAmount") < 10000, "Large")
    .otherwise("Enterprise")
)
df = df.withColumn("ETL_LoadDate", current_timestamp())
df = df.withColumn("ETL_BatchID", lit(batch_id))
```

#### 4. **Conditional Split** → DataFrame filter / when
```python
# SSIS: Route rows based on data quality conditions
# Databricks:
df_valid = df.filter(
    col("CustomerSK").isNotNull() & 
    col("ProductSK").isNotNull() & 
    col("DateSK").isNotNull() & 
    (col("Quantity") > 0) & 
    (col("NetAmount") >= 0)
)

df_missing_customer = df.filter(col("CustomerSK").isNull())
df_missing_product = df.filter(col("ProductSK").isNull())
df_bad_data = df.filter((col("Quantity") <= 0) | (col("NetAmount") < 0))
```

#### 5. **Multicast** → Cache DataFrame and reuse
```python
# SSIS: Split stream to multiple destinations
# Databricks:
df_valid.cache()  # or persist()
# Then use df_valid multiple times
```

#### 6. **Aggregate** → groupBy + agg
```python
# SSIS: Daily sales summary aggregation
# Databricks:
df_daily_summary = df_valid.groupBy("DateSK").agg(
    sum("NetAmount").alias("TotalSalesAmount"),
    sum("Quantity").alias("TotalQuantity"),
    count("*").alias("TransactionCount"),
    countDistinct("CustomerSK").alias("UniqueCustomers"),
    countDistinct("ProductSK").alias("UniqueProducts"),
    avg("NetAmount").alias("AvgTransactionValue")
)
```

#### 7. **Sort** → orderBy / sortWithinPartitions
```python
# SSIS: Sort with duplicate elimination
# Databricks:
df_sorted = df_valid.dropDuplicates(["SaleID"]).orderBy("SaleID")
```

#### 8. **Union All** → union / unionByName
```python
# SSIS: Combine error streams
# Databricks:
df_errors = df_missing_customer.union(df_missing_product).union(df_bad_data)
```

#### 9. **Row Count** → count() or Accumulator
```python
# SSIS: Track row counts in variables
# Databricks:
rows_extracted = df_sales.count()
# Or use streaming metrics for large datasets
```

#### 10. **OLE DB Destination** → Delta Lake write
```python
# SSIS: Fast Load to fact table
# Databricks:
df_valid.write.format("delta").mode("append").saveAsTable("fact.sales")
```

#### 11. **Flat File Destination** → Write to cloud storage
```python
# SSIS: Write errors to CSV
# Databricks:
df_errors.write.format("csv").mode("append").save("/mnt/logs/errors/")
```

---

## Data Flow 2: Customer SCD Type 2

### Key Components:

#### **Slowly Changing Dimension (SCD)** → Delta Lake MERGE
```python
# SSIS SCD Type 2 logic → Delta Lake MERGE with SCD handling

from delta.tables import DeltaTable

# Get the target table
target = DeltaTable.forName(spark, "dim.customer")

# Expire old records for changed historical attributes
target.alias("t").merge(
    source_df.alias("s"),
    "t.CustomerID = s.CustomerID AND t.IsCurrent = true"
).whenMatchedUpdate(
    condition="""
        t.CustomerSegment != s.CustomerSegment OR
        t.CreditLimit != s.CreditLimit OR
        t.AccountManager != s.AccountManager
    """,
    set={
        "IsCurrent": "false",
        "EffectiveEndDate": "current_timestamp()",
        "ModifiedDate": "current_timestamp()"
    }
).execute()

# Insert new records (both new customers and new versions)
new_records = source_df.withColumn("EffectiveStartDate", current_timestamp()) \
                       .withColumn("EffectiveEndDate", lit("9999-12-31").cast("timestamp")) \
                       .withColumn("IsCurrent", lit(True)) \
                       .withColumn("CreatedDate", current_timestamp())

target.alias("t").merge(
    new_records.alias("s"),
    "t.CustomerID = s.CustomerID AND t.IsCurrent = true"
).whenNotMatched().insertAll() \
 .whenMatchedUpdate(  # Type 1 changes
    condition="t.CustomerName != s.CustomerName OR t.Email != s.Email",
    set={
        "CustomerName": "s.CustomerName",
        "Email": "s.Email",
        "Phone": "s.Phone",
        "Address": "s.Address",
        "ModifiedDate": "current_timestamp()"
    }
).execute()
```

---

## Variables → Databricks Widgets/Parameters

```python
# SSIS Variables → Databricks
dbutils.widgets.text("extract_start_date", "2024-01-01")
dbutils.widgets.text("extract_end_date", "2024-01-02")

extract_start = dbutils.widgets.get("extract_start_date")
extract_end = dbutils.widgets.get("extract_end_date")

# For tracking metrics
rows_extracted = 0
rows_inserted = 0
rows_updated = 0
error_count = 0
```

---

## Connection Managers → Databricks Secrets

```python
# SSIS Connection Managers → Databricks
# Store credentials in Azure Key Vault or Databricks Secrets

jdbc_url_source = dbutils.secrets.get(scope="etl", key="source-jdbc-url")
jdbc_url_dest = dbutils.secrets.get(scope="etl", key="dest-jdbc-url")

# Or use Unity Catalog external connections
df = spark.read.table("source_catalog.sales_oltp.sales")
```

---

## Event Handlers → Try/Except + Workflow Error Handling

```python
# SSIS OnError Event Handler → Python try/except
try:
    # Main ETL logic
    run_etl_pipeline()
    log_success(batch_id, rows_extracted, rows_inserted)
    
except Exception as e:
    log_error(batch_id, str(e))
    update_batch_status(batch_id, "Failed")
    send_failure_notification(str(e))
    raise

finally:
    # Cleanup
    spark.catalog.clearCache()
```

---

## Watermark Pattern → Delta Lake

```python
# SSIS Watermark Table → Delta Lake
# Get last successful extract date
last_watermark = spark.sql("""
    SELECT COALESCE(MAX(LastExtractDate), '1900-01-01') as ExtractStartDate
    FROM etl.watermark_table
    WHERE TableName = 'Sales' AND Status = 'Success'
""").first()["ExtractStartDate"]

# Update watermark after success
spark.sql(f"""
    MERGE INTO etl.watermark_table AS target
    USING (SELECT 'Sales' as TableName, '{extract_end}' as LastExtractDate, 'Success' as Status) AS source
    ON target.TableName = source.TableName
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")
```

---

## Complete Databricks Notebook Structure

```
1. Notebook: 00_Main_Orchestrator
   - Sets parameters
   - Calls child notebooks
   - Handles errors

2. Notebook: 01_Extract_Transform_Sales
   - Extract from source
   - Lookups/joins
   - Transformations
   - Data quality splits
   - Load to fact table

3. Notebook: 02_Customer_SCD
   - SCD Type 2 processing
   - Delta Lake MERGE operations

4. Notebook: 03_Utilities
   - Logging functions
   - Notification helpers
   - Watermark management
```

---

## Key Conversion Considerations

1. **Parallelism**: SSIS data flows are single-threaded per buffer; Spark is distributed
2. **Caching**: Use `.cache()` or `.persist()` instead of SSIS buffer reuse
3. **Error handling**: SSIS row-level error handling → Spark exception handling or quarantine tables
4. **Transactions**: SSIS transactions → Delta Lake ACID transactions
5. **Incremental loads**: SSIS variables/watermarks → Delta Lake change data feed or watermark tables
6. **SCD**: SSIS SCD component → Delta Lake MERGE statements

Good luck with your conversion! 🚀
