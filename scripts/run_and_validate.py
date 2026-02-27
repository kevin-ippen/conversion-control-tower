"""
Validation Runner for Converted SSIS Package

Executes the converted ETL logic against test data and validates/scores the output.
This simulates what would happen when running the converted notebooks in Databricks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    passed: bool
    expected: Any
    actual: Any
    message: str
    severity: str = "error"  # error, warning, info


@dataclass
class QualityScore:
    """Quality score for the conversion."""
    overall_score: float
    categories: Dict[str, float] = field(default_factory=dict)
    validations: List[ValidationResult] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class ConvertedETLRunner:
    """
    Runs the converted ETL logic using pandas (simulating Spark).
    """

    def __init__(self, test_data_dir: Path):
        self.test_data_dir = test_data_dir
        self.metrics = {}

    def load_test_data(self) -> Dict[str, pd.DataFrame]:
        """Load all test data into DataFrames."""
        data = {}

        csv_dir = self.test_data_dir / "csv"

        # Load source tables
        data["sales"] = pd.read_csv(csv_dir / "sales.csv", parse_dates=["sale_date", "created_date", "modified_date"])
        data["customers"] = pd.read_csv(csv_dir / "customers.csv")

        # Load dimension tables
        data["dim_customer"] = pd.read_csv(csv_dir / "dim_customer.csv", parse_dates=["effective_start_date", "effective_end_date", "created_date", "modified_date"])
        data["dim_product"] = pd.read_csv(csv_dir / "dim_product.csv")
        data["dim_date"] = pd.read_csv(csv_dir / "dim_date.csv", parse_dates=["full_date"])

        print(f"Loaded test data:")
        for name, df in data.items():
            print(f"  - {name}: {len(df)} rows")

        return data

    def run_sales_etl(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Execute the Sales ETL logic (converted from DFT - Extract Transform Sales).
        This mirrors what the converted notebook would do.
        """
        print("\n--- Running Sales ETL Pipeline ---")
        results = {}

        # Step 1: Source extraction (OLE_SRC - Sales)
        print("Step 1: Extracting sales data...")
        df_sales = data["sales"].copy()
        df_sales["GrossAmount"] = df_sales["quantity"] * df_sales["unit_price"]
        df_sales["NetAmount"] = df_sales["GrossAmount"] * (1 - df_sales["discount_percent"].fillna(0) / 100)
        self.metrics["rows_extracted"] = len(df_sales)
        print(f"  Extracted {len(df_sales)} rows")

        # Step 2: Customer Lookup (LKP - Customer Dim)
        print("Step 2: Joining with customer dimension...")
        dim_customer_current = data["dim_customer"][data["dim_customer"]["is_current"] == True]
        df_with_customer = df_sales.merge(
            dim_customer_current[["customer_sk", "customer_id", "customer_name", "customer_segment", "region"]],
            on="customer_id",
            how="left"
        )
        missing_customers = df_with_customer["customer_sk"].isna().sum()
        print(f"  Joined {len(df_with_customer)} rows, {missing_customers} missing customer lookups")

        # Step 3: Product Lookup (LKP - Product Dim)
        print("Step 3: Joining with product dimension...")
        df_with_product = df_with_customer.merge(
            data["dim_product"][["product_sk", "product_id", "product_name", "category", "sub_category", "standard_cost"]],
            on="product_id",
            how="left"
        )
        missing_products = df_with_product["product_sk"].isna().sum()
        print(f"  Joined {len(df_with_product)} rows, {missing_products} missing product lookups")

        # Step 4: Date Lookup (LKP - Date Dim)
        print("Step 4: Joining with date dimension...")
        df_with_product["sale_date_only"] = pd.to_datetime(df_with_product["sale_date"]).dt.date
        data["dim_date"]["full_date_only"] = pd.to_datetime(data["dim_date"]["full_date"]).dt.date
        df_enriched = df_with_product.merge(
            data["dim_date"][["date_sk", "full_date_only", "fiscal_year", "fiscal_quarter", "is_weekend", "is_holiday"]],
            left_on="sale_date_only",
            right_on="full_date_only",
            how="left"
        )
        print(f"  Joined {len(df_enriched)} rows")

        # Step 5: Derived Columns (DER - Calculate Metrics)
        print("Step 5: Calculating derived metrics...")
        df_enriched["ProfitAmount"] = df_enriched["NetAmount"] - (df_enriched["quantity"] * df_enriched["standard_cost"].fillna(0))
        df_enriched["ProfitMarginPct"] = np.where(
            df_enriched["NetAmount"] == 0,
            0,
            ((df_enriched["NetAmount"] - (df_enriched["quantity"] * df_enriched["standard_cost"].fillna(0))) / df_enriched["NetAmount"]) * 100
        )
        df_enriched["SaleCategory"] = pd.cut(
            df_enriched["NetAmount"],
            bins=[-np.inf, 100, 1000, 10000, np.inf],
            labels=["Small", "Medium", "Large", "Enterprise"]
        )
        df_enriched["ETL_LoadDate"] = datetime.now()
        df_enriched["ETL_BatchID"] = 1
        print(f"  Added derived columns")

        # Step 6: Conditional Split (CSPL - Data Quality Check)
        print("Step 6: Applying data quality checks...")
        valid_mask = (
            df_enriched["customer_sk"].notna() &
            df_enriched["product_sk"].notna() &
            df_enriched["date_sk"].notna() &
            (df_enriched["quantity"] > 0) &
            (df_enriched["NetAmount"] >= 0)
        )
        df_valid = df_enriched[valid_mask].copy()
        df_invalid = df_enriched[~valid_mask].copy()

        # Categorize errors
        df_missing_customer = df_enriched[df_enriched["customer_sk"].isna()].copy()
        df_missing_product = df_enriched[df_enriched["product_sk"].isna()].copy()
        df_bad_data = df_enriched[(df_enriched["quantity"] <= 0) | (df_enriched["NetAmount"] < 0)].copy()

        self.metrics["valid_rows"] = len(df_valid)
        self.metrics["invalid_rows"] = len(df_invalid)
        self.metrics["missing_customer"] = len(df_missing_customer)
        self.metrics["missing_product"] = len(df_missing_product)
        self.metrics["bad_data"] = len(df_bad_data)

        print(f"  Valid: {len(df_valid)}, Invalid: {len(df_invalid)}")
        print(f"    - Missing customer: {len(df_missing_customer)}")
        print(f"    - Missing product: {len(df_missing_product)}")
        print(f"    - Bad data: {len(df_bad_data)}")

        # Step 7: Aggregate (AGG - Daily Sales Summary)
        print("Step 7: Creating daily sales summary...")
        df_daily_summary = df_valid.groupby("date_sk").agg(
            TotalSalesAmount=("NetAmount", "sum"),
            TotalQuantity=("quantity", "sum"),
            TransactionCount=("sale_id", "count"),
            UniqueCustomers=("customer_sk", "nunique"),
            UniqueProducts=("product_sk", "nunique"),
            AvgTransactionValue=("NetAmount", "mean")
        ).reset_index()
        print(f"  Created {len(df_daily_summary)} daily summary rows")

        # Step 8: Sort and deduplicate
        print("Step 8: Sorting and deduplicating...")
        df_fact_sales = df_valid.drop_duplicates(subset=["sale_id"]).sort_values("sale_id")
        print(f"  Final fact table: {len(df_fact_sales)} rows")

        # Step 9: Union errors
        print("Step 9: Combining error records...")
        error_frames = []
        if len(df_missing_customer) > 0:
            df_missing_customer["ErrorType"] = "Missing Customer"
            error_frames.append(df_missing_customer)
        if len(df_missing_product) > 0:
            df_missing_product["ErrorType"] = "Missing Product"
            error_frames.append(df_missing_product)
        if len(df_bad_data) > 0:
            df_bad_data["ErrorType"] = "Bad Data"
            error_frames.append(df_bad_data)

        if error_frames:
            df_errors = pd.concat(error_frames, ignore_index=True)
        else:
            df_errors = pd.DataFrame()

        self.metrics["error_count"] = len(df_errors)
        print(f"  Error log: {len(df_errors)} rows")

        results["fact_sales"] = df_fact_sales
        results["daily_summary"] = df_daily_summary
        results["errors"] = df_errors

        return results

    def run_customer_scd(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Execute Customer SCD Type 2 logic (converted from DFT - Customer SCD Type 2).
        """
        print("\n--- Running Customer SCD Pipeline ---")
        results = {}

        # For simulation, we'll show what changes would be detected
        # In real execution, this would perform Delta MERGE operations

        # Get customers from sales that might have changes
        sales_customers = data["sales"]["customer_id"].unique()
        source_customers = data["customers"][data["customers"]["customer_id"].isin(sales_customers)].copy()

        # Compare with dimension
        dim_current = data["dim_customer"][data["dim_customer"]["is_current"] == True].copy()

        # Find new customers (not in dimension)
        new_customer_ids = set(source_customers["customer_id"]) - set(dim_current["customer_id"])
        new_customers = source_customers[source_customers["customer_id"].isin(new_customer_ids)]

        # Find changed customers (Type 2 attributes differ)
        # Rename source columns before merge
        source_for_merge = source_customers.rename(columns={
            "customer_segment": "customer_segment_new",
            "credit_limit": "credit_limit_new",
            "account_manager": "account_manager_new"
        })

        merged = source_for_merge.merge(
            dim_current[["customer_id", "customer_segment", "credit_limit", "account_manager"]],
            on="customer_id",
            how="inner"
        )

        type2_changes = merged[
            (merged["customer_segment_new"] != merged["customer_segment"]) |
            (merged["credit_limit_new"] != merged["credit_limit"]) |
            (merged["account_manager_new"] != merged["account_manager"])
        ] if len(merged) > 0 else pd.DataFrame()

        self.metrics["scd_new_customers"] = len(new_customers)
        self.metrics["scd_type2_changes"] = len(type2_changes)

        print(f"  New customers to insert: {len(new_customers)}")
        print(f"  Type 2 changes detected: {len(type2_changes)}")

        results["new_customers"] = new_customers
        results["type2_changes"] = type2_changes

        return results


class OutputValidator:
    """
    Validates the ETL output and generates quality scores.
    """

    def __init__(self):
        self.validations: List[ValidationResult] = []

    def validate_fact_sales(self, fact_sales: pd.DataFrame, metrics: Dict) -> List[ValidationResult]:
        """Validate the fact sales output."""
        results = []

        # Check 1: Row count matches valid records
        expected_rows = metrics.get("valid_rows", 0)
        actual_rows = len(fact_sales)
        results.append(ValidationResult(
            check_name="fact_sales_row_count",
            passed=actual_rows == expected_rows,
            expected=expected_rows,
            actual=actual_rows,
            message=f"Fact sales row count: expected {expected_rows}, got {actual_rows}",
            severity="error" if actual_rows != expected_rows else "info"
        ))

        # Check 2: No duplicate sale IDs
        duplicate_count = fact_sales["sale_id"].duplicated().sum()
        results.append(ValidationResult(
            check_name="no_duplicate_sales",
            passed=duplicate_count == 0,
            expected=0,
            actual=duplicate_count,
            message=f"Duplicate sale IDs: {duplicate_count}",
            severity="error" if duplicate_count > 0 else "info"
        ))

        # Check 3: All surrogate keys populated
        null_customer_sk = fact_sales["customer_sk"].isna().sum()
        results.append(ValidationResult(
            check_name="customer_sk_populated",
            passed=null_customer_sk == 0,
            expected=0,
            actual=null_customer_sk,
            message=f"Null customer_sk in fact: {null_customer_sk}",
            severity="error" if null_customer_sk > 0 else "info"
        ))

        null_product_sk = fact_sales["product_sk"].isna().sum()
        results.append(ValidationResult(
            check_name="product_sk_populated",
            passed=null_product_sk == 0,
            expected=0,
            actual=null_product_sk,
            message=f"Null product_sk in fact: {null_product_sk}",
            severity="error" if null_product_sk > 0 else "info"
        ))

        # Check 4: Derived columns calculated correctly (sample check)
        if len(fact_sales) > 0:
            sample = fact_sales.iloc[0]
            expected_profit = sample["NetAmount"] - (sample["quantity"] * sample["standard_cost"])
            actual_profit = sample["ProfitAmount"]
            profit_match = abs(expected_profit - actual_profit) < 0.01
            results.append(ValidationResult(
                check_name="profit_calculation",
                passed=profit_match,
                expected=round(expected_profit, 2),
                actual=round(actual_profit, 2),
                message=f"Profit calculation check",
                severity="error" if not profit_match else "info"
            ))

        # Check 5: ETL metadata populated
        has_load_date = fact_sales["ETL_LoadDate"].notna().all()
        has_batch_id = fact_sales["ETL_BatchID"].notna().all()
        results.append(ValidationResult(
            check_name="etl_metadata_populated",
            passed=has_load_date and has_batch_id,
            expected=True,
            actual=has_load_date and has_batch_id,
            message="ETL metadata (LoadDate, BatchID) populated",
            severity="warning" if not (has_load_date and has_batch_id) else "info"
        ))

        return results

    def validate_daily_summary(self, daily_summary: pd.DataFrame, fact_sales: pd.DataFrame) -> List[ValidationResult]:
        """Validate the daily summary aggregation."""
        results = []

        # Check 1: Summary totals match fact detail
        fact_total = fact_sales["NetAmount"].sum()
        summary_total = daily_summary["TotalSalesAmount"].sum()
        totals_match = abs(fact_total - summary_total) < 0.01

        results.append(ValidationResult(
            check_name="summary_totals_match",
            passed=totals_match,
            expected=round(fact_total, 2),
            actual=round(summary_total, 2),
            message=f"Summary total matches fact total",
            severity="error" if not totals_match else "info"
        ))

        # Check 2: Transaction count matches
        fact_count = len(fact_sales)
        summary_count = daily_summary["TransactionCount"].sum()
        counts_match = fact_count == summary_count

        results.append(ValidationResult(
            check_name="transaction_counts_match",
            passed=counts_match,
            expected=fact_count,
            actual=summary_count,
            message=f"Transaction count matches",
            severity="error" if not counts_match else "info"
        ))

        return results

    def validate_error_handling(self, errors: pd.DataFrame, metrics: Dict) -> List[ValidationResult]:
        """Validate error handling."""
        results = []

        # Check: Error count matches invalid records
        expected_errors = metrics.get("invalid_rows", 0)
        # Note: Some records may have multiple error types, so this is approximate
        actual_error_records = len(errors.drop_duplicates(subset=["sale_id"])) if "sale_id" in errors.columns else 0

        results.append(ValidationResult(
            check_name="error_records_captured",
            passed=actual_error_records > 0 if expected_errors > 0 else True,
            expected=f">0 if invalid rows exist",
            actual=actual_error_records,
            message=f"Error records captured: {actual_error_records}",
            severity="warning" if expected_errors > 0 and actual_error_records == 0 else "info"
        ))

        return results

    def calculate_score(self, validations: List[ValidationResult], metrics: Dict) -> QualityScore:
        """Calculate overall quality score."""

        # Category scores
        categories = {
            "data_completeness": 0.0,
            "data_accuracy": 0.0,
            "transformation_logic": 0.0,
            "error_handling": 0.0
        }

        # Group validations by category
        completeness_checks = ["customer_sk_populated", "product_sk_populated", "etl_metadata_populated"]
        accuracy_checks = ["summary_totals_match", "transaction_counts_match"]
        logic_checks = ["profit_calculation", "fact_sales_row_count", "no_duplicate_sales"]
        error_checks = ["error_records_captured"]

        def calc_category_score(check_names: List[str]) -> float:
            relevant = [v for v in validations if v.check_name in check_names]
            if not relevant:
                return 1.0
            passed = sum(1 for v in relevant if v.passed)
            return passed / len(relevant)

        categories["data_completeness"] = calc_category_score(completeness_checks)
        categories["data_accuracy"] = calc_category_score(accuracy_checks)
        categories["transformation_logic"] = calc_category_score(logic_checks)
        categories["error_handling"] = calc_category_score(error_checks)

        # Overall score (weighted average)
        weights = {
            "data_completeness": 0.25,
            "data_accuracy": 0.30,
            "transformation_logic": 0.30,
            "error_handling": 0.15
        }

        overall = sum(categories[cat] * weights[cat] for cat in categories)

        return QualityScore(
            overall_score=overall,
            categories=categories,
            validations=validations,
            metrics=metrics
        )


def print_score_report(score: QualityScore):
    """Print a formatted score report."""
    print("\n" + "=" * 60)
    print("CONVERSION QUALITY SCORE REPORT")
    print("=" * 60)

    # Overall score with color indication
    overall_pct = score.overall_score * 100
    if overall_pct >= 90:
        grade = "A - Excellent"
    elif overall_pct >= 80:
        grade = "B - Good"
    elif overall_pct >= 70:
        grade = "C - Acceptable"
    elif overall_pct >= 60:
        grade = "D - Needs Improvement"
    else:
        grade = "F - Failed"

    print(f"\nOVERALL SCORE: {overall_pct:.1f}% ({grade})")

    print("\n--- Category Scores ---")
    for category, score_val in score.categories.items():
        bar = "█" * int(score_val * 20) + "░" * (20 - int(score_val * 20))
        print(f"  {category:25} [{bar}] {score_val*100:.0f}%")

    print("\n--- Validation Results ---")
    for v in score.validations:
        status = "✓ PASS" if v.passed else "✗ FAIL"
        print(f"  {status:10} {v.check_name}: {v.message}")

    print("\n--- ETL Metrics ---")
    for key, value in score.metrics.items():
        print(f"  {key:25} {value}")

    print("\n" + "=" * 60)


def main():
    """Main entry point for validation runner."""
    print("=" * 60)
    print("SSIS to Databricks Conversion Validator")
    print("=" * 60)

    # Paths
    project_root = Path(__file__).parent.parent
    test_data_dir = project_root / "samples" / "test_data"
    output_dir = project_root / "output" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run ETL
    runner = ConvertedETLRunner(test_data_dir)

    # Load test data
    print("\n[1/4] Loading test data...")
    data = runner.load_test_data()

    # Run Sales ETL
    print("\n[2/4] Running Sales ETL pipeline...")
    sales_results = runner.run_sales_etl(data)

    # Run Customer SCD
    print("\n[3/4] Running Customer SCD pipeline...")
    scd_results = runner.run_customer_scd(data)

    # Validate
    print("\n[4/4] Validating output...")
    validator = OutputValidator()

    all_validations = []
    all_validations.extend(validator.validate_fact_sales(sales_results["fact_sales"], runner.metrics))
    all_validations.extend(validator.validate_daily_summary(sales_results["daily_summary"], sales_results["fact_sales"]))
    all_validations.extend(validator.validate_error_handling(sales_results["errors"], runner.metrics))

    # Calculate score
    score = validator.calculate_score(all_validations, runner.metrics)

    # Print report
    print_score_report(score)

    # Save outputs
    print("\n--- Saving Output Files ---")
    sales_results["fact_sales"].to_csv(output_dir / "fact_sales.csv", index=False)
    sales_results["daily_summary"].to_csv(output_dir / "daily_summary.csv", index=False)
    if len(sales_results["errors"]) > 0:
        sales_results["errors"].to_csv(output_dir / "errors.csv", index=False)

    # Save score report
    score_report = {
        "overall_score": float(score.overall_score),
        "categories": {k: float(v) for k, v in score.categories.items()},
        "metrics": {k: int(v) if isinstance(v, (int, np.integer)) else v for k, v in score.metrics.items()},
        "validations": [
            {
                "check_name": v.check_name,
                "passed": bool(v.passed),
                "expected": str(v.expected),
                "actual": str(v.actual),
                "message": v.message,
                "severity": v.severity
            }
            for v in score.validations
        ],
        "timestamp": datetime.now().isoformat()
    }
    with open(output_dir / "validation_report.json", "w") as f:
        json.dump(score_report, f, indent=2)

    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - fact_sales.csv")
    print(f"  - daily_summary.csv")
    print(f"  - errors.csv")
    print(f"  - validation_report.json")

    return score


if __name__ == "__main__":
    main()
