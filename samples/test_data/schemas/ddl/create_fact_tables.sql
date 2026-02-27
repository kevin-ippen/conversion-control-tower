-- Fact Tables DDL for SalesDataETL Test Data
-- These are the target tables for the ETL process

CREATE TABLE IF NOT EXISTS fact_sales (
    SalesSK INT GENERATED ALWAYS AS IDENTITY,
    SaleID INT,
    DateSK INT,
    CustomerSK INT,
    ProductSK INT,
    StoreSK INT,
    SalesRepSK INT,
    Quantity INT,
    UnitPrice DECIMAL(10,2),
    DiscountPercent DOUBLE,
    GrossAmount DECIMAL(10,2),
    NetAmount DECIMAL(10,2),
    ProfitAmount DECIMAL(10,2),
    ProfitMarginPct DOUBLE,
    SaleCategory STRING,
    ETL_LoadDate TIMESTAMP,
    ETL_BatchID INT
);

CREATE TABLE IF NOT EXISTS fact_daily_sales_summary (
    DateSK INT,
    TotalSalesAmount DECIMAL(15,2),
    TotalQuantity BIGINT,
    TransactionCount INT,
    UniqueCustomers INT,
    UniqueProducts INT,
    AvgTransactionValue DECIMAL(10,2),
    ETL_LoadDate TIMESTAMP
);

CREATE TABLE IF NOT EXISTS etl_batch_log (
    BatchID INT PRIMARY KEY,
    PackageName STRING,
    BatchStartTime TIMESTAMP,
    BatchEndTime TIMESTAMP,
    Status STRING,
    RowsExtracted INT,
    RowsInserted INT,
    RowsUpdated INT,
    ErrorCount INT
);

CREATE TABLE IF NOT EXISTS etl_watermark_table (
    TableName STRING PRIMARY KEY,
    LastExtractDate TIMESTAMP,
    Status STRING,
    CreatedDate TIMESTAMP,
    ModifiedDate TIMESTAMP
);

CREATE TABLE IF NOT EXISTS etl_error_log (
    ErrorID INT GENERATED ALWAYS AS IDENTITY,
    BatchID INT,
    ErrorCode STRING,
    ErrorDescription STRING,
    SourceName STRING,
    ErrorTime TIMESTAMP
);
