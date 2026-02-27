-- Source Tables DDL for SalesDataETL Test Data
-- These simulate the SQL Server OLTP source

CREATE TABLE IF NOT EXISTS dbo_sales (
    SaleID INT PRIMARY KEY,
    CustomerID INT,
    ProductID INT,
    SaleDate TIMESTAMP,
    Quantity INT,
    UnitPrice DECIMAL(10,2),
    DiscountPercent DOUBLE,
    StoreID INT,
    SalesRepID INT,
    CreatedDate TIMESTAMP,
    ModifiedDate TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dbo_customers (
    CustomerID INT PRIMARY KEY,
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
    AccountManager STRING
);
