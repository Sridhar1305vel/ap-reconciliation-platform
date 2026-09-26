"""
Reporting Layer
-----------------
Combines the rules-based reconciliation output with the ML anomaly
scores into ONE flat, well-typed master table designed to be dropped
straight into Power BI (or Tableau/Excel) with no further cleanup.

Also computes headline KPIs a finance stakeholder actually cares about:
  - match/pass rate
  - financial exposure of flagged discrepancies (currency-normalized)
  - top vendors by discrepancy count/value
  - monthly anomaly trend

Generates:
  - reports/master_reconciliation_export.csv  (Power BI data source)
  - reports/summary_report.json               (KPIs, machine-readable)
  - reports/charts/*.png                      (headline visuals)
  - reports/SUMMARY.md                        (human-readable report)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ingest import load_and_clean_invoices, load_and_clean_payments

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
CHARTS_DIR = os.path.join(REPORTS_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# Fixed FX rates for currency-normalized exposure reporting.
# Simplification for this project - a production system would pull live rates.
FX_TO_GBP = {"GBP": 1.0, "USD": 0.79, "EUR": 0.85}

ANOMALY_STATUSES = {
    "matched_amount_mismatch", "matched_duplicate", "matched_duplicate+date_anomaly",
    "date_anomaly", "unmatched_payment",
}


def build_master_table() -> pd.DataFrame:
    recon = pd.read_csv(os.path.join(REPORTS_DIR, "reconciliation_results.csv"))
    ml = pd.read_csv(os.path.join(REPORTS_DIR, "ml_predictions.csv"))
    invoices, _ = load_and_clean_invoices()
    payments, _ = load_and_clean_payments()

    master = recon.merge(
        ml[["invoice_id", "transaction_id", "isolation_forest_risk", "isolation_forest_flag",
            "random_forest_risk", "random_forest_flag", "combined_flag", "is_anomaly"]],
        on=["invoice_id", "transaction_id"], how="left",
    )

    master = master.merge(
        invoices[["invoice_id", "invoice_date", "due_date", "currency"]],
        on="invoice_id", how="left",
    )
    master = master.merge(
        payments[["transaction_id", "payment_date", "currency"]],
        on="transaction_id", how="left", suffixes=("", "_pay"),
    )
    master["currency"] = master["currency"].fillna(master["currency_pay"])
    master.drop(columns=["currency_pay"], inplace=True)

    master["fx_rate_to_gbp"] = master["currency"].map(FX_TO_GBP).fillna(1.0)
    master["exposure_gbp"] = (
        master["invoice_amount"].fillna(master["payment_amount"]).fillna(0) * master["fx_rate_to_gbp"]
    )

    def risk_level(row):
        if row["status"] == "unmatched_payment" or row["random_forest_risk"] >= 0.75:
            return "High"
        if row["combined_flag"] == 1:
            return "Medium"
        return "Low"

    master["risk_level"] = master.apply(risk_level, axis=1)
    master["is_rules_flagged"] = master["status"].isin(ANOMALY_STATUSES).astype(int)

    ordered_cols = [
        "invoice_id", "transaction_id", "vendor_name", "currency",
        "invoice_date", "payment_date", "due_date",
        "invoice_amount", "payment_amount", "amount_pct_diff", "date_lag_days",
        "match_method", "match_confidence", "status",
        "is_rules_flagged", "isolation_forest_risk", "isolation_forest_flag",
        "random_forest_risk", "random_forest_flag", "combined_flag", "risk_level",
        "exposure_gbp", "is_anomaly",
    ]
    master = master[ordered_cols].rename(columns={"is_anomaly": "ground_truth_anomaly_label"})
    return master


def compute_kpis(master: pd.DataFrame, invoices: pd.DataFrame) -> dict:
    total_invoices = len(invoices)
    invoice_level = master[master["invoice_id"].notna()].drop_duplicates(subset="invoice_id", keep="first")
    # For pass-rate purposes, an invoice counts as "clean" if ANY of its rows is a clean/fuzzy match
    clean_ids = set(master[master["status"].isin(["matched_clean", "matched_fuzzy"])]["invoice_id"].dropna())
    pass_rate = round(len(clean_ids) / total_invoices, 4)

    status_counts = master["status"].value_counts().to_dict()

    flagged = master[master["combined_flag"] == 1]
    exposure_by_status = (
        master[master["is_rules_flagged"] == 1]
        .groupby("status")["exposure_gbp"].sum().round(2).sort_values(ascending=False).to_dict()
    )
    total_exposure_gbp = round(master.loc[master["is_rules_flagged"] == 1, "exposure_gbp"].sum(), 2)

    top_vendors = (
        master[master["is_rules_flagged"] == 1]
        .groupby("vendor_name")
        .agg(discrepancy_count=("vendor_name", "count"), exposure_gbp=("exposure_gbp", "sum"))
        .sort_values("exposure_gbp", ascending=False)
        .head(5)
        .round(2)
        .reset_index()
        .to_dict(orient="records")
    )

    master["month"] = pd.to_datetime(master["invoice_date"]).dt.to_period("M").astype(str)
    monthly_trend = (
        master[master["is_rules_flagged"] == 1]
        .groupby("month").size().sort_index().to_dict()
    )

    risk_level_counts = master["risk_level"].value_counts().to_dict()

    return {
        "total_invoices": total_invoices,
        "total_payments": len(pd.read_csv(os.path.join(PROJECT_ROOT, "data", "payments.csv"))),
        "pass_rate": pass_rate,
        "status_counts": status_counts,
        "risk_level_counts": risk_level_counts,
        "total_flagged_records": int(len(flagged)),
        "total_financial_exposure_gbp": total_exposure_gbp,
        "exposure_by_status_gbp": exposure_by_status,
        "top_vendors_by_exposure": top_vendors,
        "monthly_anomaly_trend": monthly_trend,
    }


def generate_charts(master: pd.DataFrame, kpis: dict):
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Status distribution donut
    fig, ax = plt.subplots(figsize=(7, 6))
    status_counts = pd.Series(kpis["status_counts"]).sort_values(ascending=False)
    colors = plt.cm.Set2.colors
    ax.pie(status_counts.values, labels=status_counts.index, autopct="%1.1f%%",
           startangle=90, colors=colors, wedgeprops=dict(width=0.4))
    ax.set_title("Reconciliation Status Distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "status_distribution.png"), dpi=150)
    plt.close(fig)

    # 2. Financial exposure by status
    fig, ax = plt.subplots(figsize=(8, 5))
    exp = pd.Series(kpis["exposure_by_status_gbp"]).sort_values()
    ax.barh(exp.index, exp.values, color="#c0392b")
    ax.set_xlabel("Exposure (GBP)")
    ax.set_title("Financial Exposure by Discrepancy Type")
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "exposure_by_status.png"), dpi=150)
    plt.close(fig)

    # 3. Monthly anomaly trend
    fig, ax = plt.subplots(figsize=(8, 5))
    trend = pd.Series(kpis["monthly_anomaly_trend"]).sort_index()
    ax.plot(trend.index, trend.values, marker="o", color="#2980b9")
    ax.set_title("Monthly Flagged-Discrepancy Trend")
    ax.set_ylabel("Number of flagged records")
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "monthly_trend.png"), dpi=150)
    plt.close(fig)

    # 4. Top vendors by exposure
    fig, ax = plt.subplots(figsize=(8, 5))
    tv = pd.DataFrame(kpis["top_vendors_by_exposure"])
    if not tv.empty:
        ax.barh(tv["vendor_name"], tv["exposure_gbp"], color="#8e44ad")
        ax.invert_yaxis()
        ax.set_xlabel("Exposure (GBP)")
        ax.set_title("Top 5 Vendors by Flagged Exposure")
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "top_vendors.png"), dpi=150)
    plt.close(fig)


def write_summary_md(kpis: dict):
    lines = [
        "# Reconciliation Summary Report\n",
        f"**Total invoices:** {kpis['total_invoices']}  ",
        f"**Total payments:** {kpis['total_payments']}  ",
        f"**Pass rate (clean/fuzzy matched):** {kpis['pass_rate']*100:.1f}%  ",
        f"**Total flagged records:** {kpis['total_flagged_records']}  ",
        f"**Total financial exposure flagged:** £{kpis['total_financial_exposure_gbp']:,.2f}\n",
        "## Status Breakdown\n",
    ]
    for status, count in kpis["status_counts"].items():
        lines.append(f"- {status}: {count}")

    lines += ["\n## Financial Exposure by Discrepancy Type (GBP)\n"]
    for status, amt in kpis["exposure_by_status_gbp"].items():
        lines.append(f"- {status}: £{amt:,.2f}")

    lines += ["\n## Top Vendors by Flagged Exposure\n"]
    for v in kpis["top_vendors_by_exposure"]:
        lines.append(f"- {v['vendor_name']}: {v['discrepancy_count']} discrepancies, £{v['exposure_gbp']:,.2f}")

    lines += [
        "\n## Charts\n",
        "![Status Distribution](charts/status_distribution.png)",
        "![Exposure by Status](charts/exposure_by_status.png)",
        "![Monthly Trend](charts/monthly_trend.png)",
        "![Top Vendors](charts/top_vendors.png)",
    ]

    with open(os.path.join(REPORTS_DIR, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run():
    invoices, _ = load_and_clean_invoices()
    master = build_master_table()
    master.to_csv(os.path.join(REPORTS_DIR, "master_reconciliation_export.csv"), index=False)

    kpis = compute_kpis(master, invoices)
    with open(os.path.join(REPORTS_DIR, "summary_report.json"), "w", encoding="utf-8") as f:
        json.dump(kpis, f, indent=2, default=str)

    generate_charts(master, kpis)
    write_summary_md(kpis)

    print("=== RECONCILIATION SUMMARY ===")
    print(f"Pass rate: {kpis['pass_rate']*100:.1f}%")
    print(f"Total flagged records: {kpis['total_flagged_records']}")
    print(f"Total financial exposure flagged: £{kpis['total_financial_exposure_gbp']:,.2f}")
    print("\nTop vendors by exposure:")
    for v in kpis["top_vendors_by_exposure"]:
        print(f"  {v['vendor_name']}: £{v['exposure_gbp']:,.2f} across {v['discrepancy_count']} records")
    print(f"\nSaved -> master_reconciliation_export.csv, summary_report.json, SUMMARY.md, charts/*.png")

    return master, kpis


if __name__ == "__main__":
    run()
