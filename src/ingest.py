"""
Ingestion & Cleaning Pipeline
------------------------------
Loads raw invoices.csv and payments.csv, standardizes types/formatting,
and runs structural validation BEFORE any reconciliation logic runs.

This mirrors real AP-automation systems: you never reconcile on raw,
unvalidated data. Bad rows are quarantined and reported, not silently
dropped or silently matched.

Produces:
  - cleaned in-memory DataFrames (used by the reconciliation engine)
  - reports/ingestion_quality_report.json  (machine-readable)
  - printed summary (human-readable)
"""

import json
import os
import re
from dataclasses import dataclass, field

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

VALID_CURRENCIES = {"GBP", "USD", "EUR"}


@dataclass
class QualityReport:
    """Accumulates validation findings for one table."""
    table_name: str
    total_rows: int = 0
    issues: dict = field(default_factory=dict)  # issue_type -> count
    flagged_row_ids: dict = field(default_factory=dict)  # issue_type -> [ids]

    def flag(self, issue_type: str, row_id):
        self.issues[issue_type] = self.issues.get(issue_type, 0) + 1
        self.flagged_row_ids.setdefault(issue_type, []).append(row_id)

    def to_dict(self):
        return {
            "table": self.table_name,
            "total_rows": self.total_rows,
            "issues": self.issues,
            "flagged_row_ids": self.flagged_row_ids,
        }


def _normalize_vendor(name: str) -> str:
    """Lowercase, strip punctuation/whitespace/legal suffixes for matching purposes.
    The ORIGINAL vendor_name is preserved separately for display/reporting."""
    if pd.isna(name):
        return ""
    name = name.lower().strip()
    name = re.sub(r"[.,]", "", name)
    name = re.sub(r"\b(ltd|llp|inc|plc|and)\b", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def load_and_clean_invoices(path: str = None) -> tuple[pd.DataFrame, QualityReport]:
    path = path or os.path.join(DATA_DIR, "invoices.csv")
    df = pd.read_csv(path)
    report = QualityReport(table_name="invoices", total_rows=len(df))

    # Type coercion
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["due_date"] = pd.to_datetime(df["due_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["currency"] = df["currency"].astype(str).str.upper().str.strip()
    df["vendor_name"] = df["vendor_name"].astype(str).str.strip()
    df["vendor_key"] = df["vendor_name"].apply(_normalize_vendor)

    # Validation
    for idx, row in df.iterrows():
        if pd.isna(row["invoice_date"]) or pd.isna(row["due_date"]):
            report.flag("invalid_date", row["invoice_id"])
        if pd.isna(row["amount"]) or row["amount"] <= 0:
            report.flag("invalid_amount", row["invoice_id"])
        if row["currency"] not in VALID_CURRENCIES:
            report.flag("invalid_currency", row["invoice_id"])
        if not row["vendor_key"]:
            report.flag("missing_vendor", row["invoice_id"])

    dupes = df[df.duplicated(subset="invoice_id", keep=False)]
    for inv_id in dupes["invoice_id"].unique():
        report.flag("duplicate_invoice_id", inv_id)

    return df, report


def load_and_clean_payments(path: str = None) -> tuple[pd.DataFrame, QualityReport]:
    path = path or os.path.join(DATA_DIR, "payments.csv")
    df = pd.read_csv(path)
    report = QualityReport(table_name="payments", total_rows=len(df))

    df["payment_date"] = pd.to_datetime(df["payment_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["currency"] = df["currency"].astype(str).str.upper().str.strip()
    df["vendor_name"] = df["vendor_name"].astype(str).str.strip()
    df["vendor_key"] = df["vendor_name"].apply(_normalize_vendor)

    # Extract referenced invoice_id from the free-text reference_note, e.g.
    # "Payment for INV-100747" -> "INV-100747". This is a common real-world
    # pattern: the ERP note field is unstructured, so we parse it defensively.
    df["referenced_invoice_id"] = df["reference_note"].str.extract(r"(INV-\d+)")

    for idx, row in df.iterrows():
        if pd.isna(row["payment_date"]):
            report.flag("invalid_date", row["transaction_id"])
        if pd.isna(row["amount"]) or row["amount"] <= 0:
            report.flag("invalid_amount", row["transaction_id"])
        if row["currency"] not in VALID_CURRENCIES:
            report.flag("invalid_currency", row["transaction_id"])
        if not row["vendor_key"]:
            report.flag("missing_vendor", row["transaction_id"])
        if pd.isna(row["referenced_invoice_id"]):
            report.flag("unparseable_reference", row["transaction_id"])

    dupes = df[df.duplicated(subset="transaction_id", keep=False)]
    for txn_id in dupes["transaction_id"].unique():
        report.flag("duplicate_transaction_id", txn_id)

    return df, report


def run(save_report: bool = True):
    invoices_df, inv_report = load_and_clean_invoices()
    payments_df, pay_report = load_and_clean_payments()

    combined = {
        "invoices": inv_report.to_dict(),
        "payments": pay_report.to_dict(),
    }

    if save_report:
        out_path = os.path.join(REPORTS_DIR, "ingestion_quality_report.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, default=str)
        print(f"Saved ingestion quality report -> {out_path}")

    print("\n=== INGESTION SUMMARY ===")
    for name, rep in [("Invoices", inv_report), ("Payments", pay_report)]:
        print(f"\n{name}: {rep.total_rows} rows")
        if rep.issues:
            for issue, count in rep.issues.items():
                print(f"   - {issue}: {count}")
        else:
            print("   - no structural issues found")

    return invoices_df, payments_df, inv_report, pay_report


if __name__ == "__main__":
    run()
