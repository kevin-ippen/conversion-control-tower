"""
Synthetic Test Data Generator for SalesDataETL SSIS Package

Generates test data matching the schema from the SalesDataETL.dtsx package
for validation testing of the converted Databricks pipelines.
"""

import random
import csv
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import json

# Try to import pandas/pyarrow for parquet, fall back to CSV only
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not installed, will only generate CSV files")


# Configuration
RANDOM_SEED = 42
NUM_CUSTOMERS = 50
NUM_PRODUCTS = 30
NUM_SALES = 1000
DATE_START = datetime(2024, 1, 1)
DATE_END = datetime(2024, 12, 31)

# Sample data pools
FIRST_NAMES = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
               "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
               "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
              "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
              "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson"]
CITIES = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio",
          "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville", "Fort Worth", "Columbus",
          "Charlotte", "Seattle", "Denver", "Boston", "Portland", "Miami"]
STATES = ["NY", "CA", "IL", "TX", "AZ", "PA", "TX", "CA", "TX", "CA", "TX", "FL", "TX", "OH",
          "NC", "WA", "CO", "MA", "OR", "FL"]
CUSTOMER_SEGMENTS = ["Premium", "Standard", "Basic"]
ACCOUNT_MANAGERS = ["Alice Chen", "Bob Martinez", "Carol White", "David Kim", "Eva Rodriguez"]

PRODUCT_CATEGORIES = {
    "Electronics": ["Laptops", "Phones", "Tablets", "Accessories"],
    "Clothing": ["Shirts", "Pants", "Dresses", "Outerwear"],
    "Home": ["Furniture", "Decor", "Kitchen", "Bedding"],
    "Sports": ["Equipment", "Apparel", "Footwear", "Accessories"]
}
BRANDS = ["TechPro", "StyleMax", "HomeComfort", "SportElite", "ValueChoice", "PremiumLine"]

# US Federal Holidays 2024
HOLIDAYS_2024 = [
    datetime(2024, 1, 1),   # New Year's Day
    datetime(2024, 1, 15),  # MLK Day
    datetime(2024, 2, 19),  # Presidents Day
    datetime(2024, 5, 27),  # Memorial Day
    datetime(2024, 6, 19),  # Juneteenth
    datetime(2024, 7, 4),   # Independence Day
    datetime(2024, 9, 2),   # Labor Day
    datetime(2024, 10, 14), # Columbus Day
    datetime(2024, 11, 11), # Veterans Day
    datetime(2024, 11, 28), # Thanksgiving
    datetime(2024, 12, 25), # Christmas
]


@dataclass
class Customer:
    customer_id: int
    customer_name: str
    email: str
    phone: str
    address: str
    city: str
    state: str
    postal_code: str
    country: str
    customer_segment: str
    credit_limit: float
    account_manager: str


@dataclass
class DimCustomer:
    customer_sk: int
    customer_id: int
    customer_name: str
    email: str
    phone: str
    address: str
    city: str
    state: str
    postal_code: str
    country: str
    customer_segment: str
    credit_limit: float
    account_manager: str
    region: str
    is_current: bool
    effective_start_date: datetime
    effective_end_date: datetime
    created_date: datetime
    modified_date: Optional[datetime]


@dataclass
class Product:
    product_id: int
    product_name: str
    category: str
    sub_category: str
    brand: str
    standard_cost: float


@dataclass
class DimProduct:
    product_sk: int
    product_id: int
    product_name: str
    category: str
    sub_category: str
    brand: str
    standard_cost: float
    is_current: bool


@dataclass
class DimDate:
    date_sk: int
    full_date: datetime
    day_of_week: int
    day_name: str
    month_number: int
    month_name: str
    quarter: int
    year: int
    fiscal_year: int
    fiscal_quarter: int
    is_weekend: bool
    is_holiday: bool


@dataclass
class Sale:
    sale_id: int
    customer_id: int
    product_id: int
    sale_date: datetime
    quantity: int
    unit_price: float
    discount_percent: Optional[float]
    store_id: int
    sales_rep_id: int
    created_date: datetime
    modified_date: datetime


def generate_phone():
    """Generate a random US phone number."""
    return f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}"


def generate_customers(n: int) -> List[Customer]:
    """Generate n random customers."""
    customers = []
    for i in range(1, n + 1):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        city_idx = random.randint(0, len(CITIES) - 1)

        customer = Customer(
            customer_id=i,
            customer_name=f"{first} {last}",
            email=f"{first.lower()}.{last.lower()}{i}@email.com",
            phone=generate_phone(),
            address=f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'Maple', 'Cedar', 'Pine'])} {random.choice(['St', 'Ave', 'Blvd', 'Dr'])}",
            city=CITIES[city_idx],
            state=STATES[city_idx],
            postal_code=f"{random.randint(10000, 99999)}",
            country="USA",
            customer_segment=random.choice(CUSTOMER_SEGMENTS),
            credit_limit=random.choice([5000, 10000, 25000, 50000, 100000]),
            account_manager=random.choice(ACCOUNT_MANAGERS)
        )
        customers.append(customer)
    return customers


def generate_dim_customers(customers: List[Customer]) -> List[DimCustomer]:
    """Generate dimension customers with some SCD Type 2 history."""
    dim_customers = []
    sk = 1
    base_date = DATE_START - timedelta(days=365)

    for cust in customers:
        # Determine region from state
        east_states = ["NY", "PA", "FL", "MA", "NC", "OH"]
        west_states = ["CA", "WA", "OR", "AZ", "CO"]
        if cust.state in east_states:
            region = "East"
        elif cust.state in west_states:
            region = "West"
        else:
            region = "Central"

        # 20% of customers have history (changed segment)
        if random.random() < 0.2:
            # Old record
            old_segment = random.choice([s for s in CUSTOMER_SEGMENTS if s != cust.customer_segment])
            dim_customers.append(DimCustomer(
                customer_sk=sk,
                customer_id=cust.customer_id,
                customer_name=cust.customer_name,
                email=cust.email,
                phone=cust.phone,
                address=cust.address,
                city=cust.city,
                state=cust.state,
                postal_code=cust.postal_code,
                country=cust.country,
                customer_segment=old_segment,
                credit_limit=cust.credit_limit * 0.8,
                account_manager=cust.account_manager,
                region=region,
                is_current=False,
                effective_start_date=base_date,
                effective_end_date=DATE_START - timedelta(days=1),
                created_date=base_date,
                modified_date=DATE_START - timedelta(days=1)
            ))
            sk += 1

        # Current record
        dim_customers.append(DimCustomer(
            customer_sk=sk,
            customer_id=cust.customer_id,
            customer_name=cust.customer_name,
            email=cust.email,
            phone=cust.phone,
            address=cust.address,
            city=cust.city,
            state=cust.state,
            postal_code=cust.postal_code,
            country=cust.country,
            customer_segment=cust.customer_segment,
            credit_limit=cust.credit_limit,
            account_manager=cust.account_manager,
            region=region,
            is_current=True,
            effective_start_date=DATE_START,
            effective_end_date=datetime(9999, 12, 31),
            created_date=DATE_START,
            modified_date=None
        ))
        sk += 1

    return dim_customers


def generate_products(n: int) -> List[Product]:
    """Generate n random products."""
    products = []
    product_id = 1

    for category, subcategories in PRODUCT_CATEGORIES.items():
        for subcat in subcategories:
            for _ in range(n // (len(PRODUCT_CATEGORIES) * 4) + 1):
                if product_id > n:
                    break
                product = Product(
                    product_id=product_id,
                    product_name=f"{random.choice(BRANDS)} {subcat[:-1] if subcat.endswith('s') else subcat} {product_id:03d}",
                    category=category,
                    sub_category=subcat,
                    brand=random.choice(BRANDS),
                    standard_cost=round(random.uniform(5, 500), 2)
                )
                products.append(product)
                product_id += 1
            if product_id > n:
                break
        if product_id > n:
            break

    return products[:n]


def generate_dim_products(products: List[Product]) -> List[DimProduct]:
    """Generate dimension products."""
    return [
        DimProduct(
            product_sk=i + 1,
            product_id=p.product_id,
            product_name=p.product_name,
            category=p.category,
            sub_category=p.sub_category,
            brand=p.brand,
            standard_cost=p.standard_cost,
            is_current=True
        )
        for i, p in enumerate(products)
    ]


def generate_dim_date(start: datetime, end: datetime) -> List[DimDate]:
    """Generate date dimension for date range."""
    dates = []
    current = start
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    month_names = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

    while current <= end:
        is_holiday = current.replace(hour=0, minute=0, second=0, microsecond=0) in HOLIDAYS_2024

        dates.append(DimDate(
            date_sk=int(current.strftime("%Y%m%d")),
            full_date=current,
            day_of_week=current.isoweekday(),
            day_name=day_names[current.weekday()],
            month_number=current.month,
            month_name=month_names[current.month - 1],
            quarter=(current.month - 1) // 3 + 1,
            year=current.year,
            fiscal_year=current.year if current.month >= 7 else current.year - 1,
            fiscal_quarter=((current.month - 7) % 12) // 3 + 1 if current.month >= 7 else ((current.month + 5) // 3),
            is_weekend=current.weekday() >= 5,
            is_holiday=is_holiday
        ))
        current += timedelta(days=1)

    return dates


def generate_sales(n: int, customers: List[Customer], products: List[Product],
                   start_date: datetime, end_date: datetime) -> List[Sale]:
    """Generate n sales transactions."""
    sales = []
    num_days = (end_date - start_date).days

    for i in range(1, n + 1):
        customer = random.choice(customers)
        product = random.choice(products)

        # Random date in range
        sale_date = start_date + timedelta(days=random.randint(0, num_days))

        # Unit price with some markup from standard cost
        unit_price = round(product.standard_cost * random.uniform(1.2, 2.5), 2)

        # 10% chance of null discount, otherwise 0-30%
        discount = None if random.random() < 0.1 else round(random.uniform(0, 30), 2)

        # 2% chance of bad data (negative quantity or invalid)
        if random.random() < 0.02:
            quantity = random.choice([-1, 0, -5])
        else:
            quantity = random.randint(1, 20)

        created_date = sale_date
        modified_date = sale_date + timedelta(hours=random.randint(0, 48))

        sales.append(Sale(
            sale_id=i,
            customer_id=customer.customer_id,
            product_id=product.product_id,
            sale_date=sale_date,
            quantity=quantity,
            unit_price=unit_price,
            discount_percent=discount,
            store_id=random.randint(1, 10),
            sales_rep_id=random.randint(1, 20),
            created_date=created_date,
            modified_date=modified_date
        ))

    # Add some sales with customer IDs that won't match dim (edge case)
    for i in range(n + 1, n + 11):
        product = random.choice(products)
        sale_date = start_date + timedelta(days=random.randint(0, num_days))

        sales.append(Sale(
            sale_id=i,
            customer_id=9999 + i,  # Non-existent customer
            product_id=product.product_id,
            sale_date=sale_date,
            quantity=random.randint(1, 5),
            unit_price=round(product.standard_cost * 1.5, 2),
            discount_percent=None,
            store_id=random.randint(1, 10),
            sales_rep_id=random.randint(1, 20),
            created_date=sale_date,
            modified_date=sale_date
        ))

    return sales


def write_csv(data: List, filepath: Path, fieldnames: List[str]):
    """Write data to CSV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in data:
            row = {}
            for field in fieldnames:
                value = getattr(item, field)
                if isinstance(value, datetime):
                    value = value.isoformat()
                elif value is None:
                    value = ''
                row[field] = value
            writer.writerow(row)
    print(f"  Written {len(data)} rows to {filepath}")


def write_parquet(data: List, filepath: Path, fieldnames: List[str]):
    """Write data to Parquet file using pandas."""
    if not HAS_PANDAS:
        return

    filepath.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in data:
        row = {}
        for field in fieldnames:
            value = getattr(item, field)
            row[field] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(filepath, index=False)
    print(f"  Written {len(data)} rows to {filepath}")


def generate_ddl_scripts(output_dir: Path):
    """Generate SQL DDL scripts for table creation."""
    ddl_dir = output_dir / "schemas" / "ddl"
    ddl_dir.mkdir(parents=True, exist_ok=True)

    # Source tables DDL
    source_ddl = """-- Source Tables DDL for SalesDataETL Test Data
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
"""
    with open(ddl_dir / "create_source_tables.sql", 'w') as f:
        f.write(source_ddl)

    # Dimension tables DDL
    dim_ddl = """-- Dimension Tables DDL for SalesDataETL Test Data
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
"""
    with open(ddl_dir / "create_dim_tables.sql", 'w') as f:
        f.write(dim_ddl)

    # Fact tables DDL
    fact_ddl = """-- Fact Tables DDL for SalesDataETL Test Data
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
"""
    with open(ddl_dir / "create_fact_tables.sql", 'w') as f:
        f.write(fact_ddl)

    print(f"  Written DDL scripts to {ddl_dir}")


def main():
    """Generate all test data."""
    random.seed(RANDOM_SEED)

    output_dir = Path(__file__).parent.parent / "samples" / "test_data"
    csv_dir = output_dir / "csv"
    parquet_dir = output_dir / "parquet"

    print(f"Generating test data...")
    print(f"  Customers: {NUM_CUSTOMERS}")
    print(f"  Products: {NUM_PRODUCTS}")
    print(f"  Sales: {NUM_SALES}")
    print(f"  Date range: {DATE_START.date()} to {DATE_END.date()}")
    print()

    # Generate data
    print("Generating customers...")
    customers = generate_customers(NUM_CUSTOMERS)
    dim_customers = generate_dim_customers(customers)

    print("Generating products...")
    products = generate_products(NUM_PRODUCTS)
    dim_products = generate_dim_products(products)

    print("Generating date dimension...")
    dim_dates = generate_dim_date(DATE_START, DATE_END)

    print("Generating sales...")
    sales = generate_sales(NUM_SALES, customers, products, DATE_START, DATE_END)

    print()
    print("Writing CSV files...")

    # Customer fields
    customer_fields = ['customer_id', 'customer_name', 'email', 'phone', 'address',
                       'city', 'state', 'postal_code', 'country', 'customer_segment',
                       'credit_limit', 'account_manager']
    write_csv(customers, csv_dir / "customers.csv", customer_fields)

    dim_customer_fields = ['customer_sk', 'customer_id', 'customer_name', 'email', 'phone',
                           'address', 'city', 'state', 'postal_code', 'country',
                           'customer_segment', 'credit_limit', 'account_manager', 'region',
                           'is_current', 'effective_start_date', 'effective_end_date',
                           'created_date', 'modified_date']
    write_csv(dim_customers, csv_dir / "dim_customer.csv", dim_customer_fields)

    # Product fields
    product_fields = ['product_id', 'product_name', 'category', 'sub_category', 'brand', 'standard_cost']
    write_csv(products, csv_dir / "products.csv", product_fields)

    dim_product_fields = ['product_sk', 'product_id', 'product_name', 'category', 'sub_category',
                          'brand', 'standard_cost', 'is_current']
    write_csv(dim_products, csv_dir / "dim_product.csv", dim_product_fields)

    # Date fields
    date_fields = ['date_sk', 'full_date', 'day_of_week', 'day_name', 'month_number',
                   'month_name', 'quarter', 'year', 'fiscal_year', 'fiscal_quarter',
                   'is_weekend', 'is_holiday']
    write_csv(dim_dates, csv_dir / "dim_date.csv", date_fields)

    # Sales fields
    sales_fields = ['sale_id', 'customer_id', 'product_id', 'sale_date', 'quantity',
                    'unit_price', 'discount_percent', 'store_id', 'sales_rep_id',
                    'created_date', 'modified_date']
    write_csv(sales, csv_dir / "sales.csv", sales_fields)

    if HAS_PANDAS:
        print()
        print("Writing Parquet files...")
        write_parquet(customers, parquet_dir / "customers.parquet", customer_fields)
        write_parquet(dim_customers, parquet_dir / "dim_customer.parquet", dim_customer_fields)
        write_parquet(products, parquet_dir / "products.parquet", product_fields)
        write_parquet(dim_products, parquet_dir / "dim_product.parquet", dim_product_fields)
        write_parquet(dim_dates, parquet_dir / "dim_date.parquet", date_fields)
        write_parquet(sales, parquet_dir / "sales.parquet", sales_fields)

    print()
    print("Generating DDL scripts...")
    generate_ddl_scripts(output_dir)

    # Write manifest
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "seed": RANDOM_SEED,
        "counts": {
            "customers": len(customers),
            "dim_customers": len(dim_customers),
            "products": len(products),
            "dim_products": len(dim_products),
            "dim_dates": len(dim_dates),
            "sales": len(sales)
        },
        "date_range": {
            "start": DATE_START.isoformat(),
            "end": DATE_END.isoformat()
        },
        "edge_cases": {
            "null_discounts": sum(1 for s in sales if s.discount_percent is None),
            "bad_quantities": sum(1 for s in sales if s.quantity <= 0),
            "missing_customers": sum(1 for s in sales if s.customer_id > NUM_CUSTOMERS)
        }
    }
    with open(output_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    print()
    print("Done! Test data generated at:")
    print(f"  {output_dir}")
    print()
    print("Summary:")
    print(f"  - {len(customers)} customers ({len(dim_customers)} dimension records)")
    print(f"  - {len(products)} products")
    print(f"  - {len(dim_dates)} dates")
    print(f"  - {len(sales)} sales")
    print(f"  - Edge cases:")
    print(f"    - {manifest['edge_cases']['null_discounts']} null discounts")
    print(f"    - {manifest['edge_cases']['bad_quantities']} bad quantities (<=0)")
    print(f"    - {manifest['edge_cases']['missing_customers']} missing customer lookups")


if __name__ == '__main__':
    main()
