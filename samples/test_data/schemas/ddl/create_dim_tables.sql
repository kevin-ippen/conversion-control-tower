-- Dimension Tables DDL for SalesDataETL Test Data
-- These simulate the Data Warehouse dimension tables

CREATE TABLE IF NOT EXISTS dim_customer (
    CustomerSK INT PRIMARY KEY,
    CustomerID INT,
    CustomerName STRING,
    Email STRING,
    Phone STRING,
    Address STRING,
    City STRING,
    State STRING,
    PostalCode STRING,
    Country STRING,
    CustomerSegment STRING,
    CreditLimit DECIMAL(10,2),
    AccountManager STRING,
    Region STRING,
    IsCurrent BOOLEAN,
    EffectiveStartDate TIMESTAMP,
    EffectiveEndDate TIMESTAMP,
    CreatedDate TIMESTAMP,
    ModifiedDate TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dim_product (
    ProductSK INT PRIMARY KEY,
    ProductID INT,
    ProductName STRING,
    Category STRING,
    SubCategory STRING,
    Brand STRING,
    StandardCost DECIMAL(10,2),
    IsCurrent BOOLEAN
);

CREATE TABLE IF NOT EXISTS dim_date (
    DateSK INT PRIMARY KEY,
    FullDate DATE,
    DayOfWeek INT,
    DayName STRING,
    MonthNumber INT,
    MonthName STRING,
    Quarter INT,
    Year INT,
    FiscalYear INT,
    FiscalQuarter INT,
    IsWeekend BOOLEAN,
    IsHoliday BOOLEAN
);
