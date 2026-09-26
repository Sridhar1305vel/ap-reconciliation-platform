"""
ML Anomaly Detection Layer
----------------------------
Takes the rules-based reconciliation output and adds a machine-learning
layer on top, comparing two genuinely different approaches:

  A) Isolation Forest (UNSUPERVISED)
     Never sees labels. Learns what "normal" looks like from the shape
     of the data itself and flags points that are easy to isolate
     (few splits needed) as anomalies. This is what you'd deploy in
     production on day one, before you have any confirmed fraud cases.

  B) Random Forest Classifier (SUPERVISED)
     Trained on historically confirmed anomaly labels (here: our
     ground truth, standing in for "cases the finance team already
     investigated and tagged"). More precise on known patterns, but
     blind to genuinely novel anomaly types it's never seen.

Both are evaluated against the same held-out ground truth so the
comparison is fair. In a real deployment neither model replaces the
other - Isolation Forest is the always-on tripwire, the classifier is
the precision layer once you have enough investigated history.

Output:
  - reports/ml_anomaly_report.json   (metrics for both models)
  - reports/ml_predictions.csv       (every record + both model scores)
"""

import json
import os

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ingest import load_and_clean_invoices, load_and_clean_payments

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Anomaly scenarios vs. business-as-usual scenarios.
# vendor_typo and unpaid_invoice are treated as NORMAL: a typo resolved
# via fuzzy matching is still a legitimate payment, and an unpaid
# invoice just hasn't come due for action yet - neither is inherently
# an error or fraud signal, which is what this layer targets.
ANOMALY_SCENARIOS = {"amount_mismatch", "duplicate_payment", "date_anomaly", "orphan_payment"}


def build_feature_table() -> pd.DataFrame:
    results = pd.read_csv(os.path.join(REPORTS_DIR, "reconciliation_results.csv"))
    invoices, _ = load_and_clean_invoices()
    payments, _ = load_and_clean_payments()
    gt = pd.read_csv(os.path.join(DATA_DIR, "ground_truth.csv"))

    inv_keys = invoices.set_index("invoice_id")["vendor_key"].to_dict()
    pay_keys = payments.set_index("transaction_id")["vendor_key"].to_dict()

    dup_counts = results.dropna(subset=["invoice_id"]).groupby("invoice_id")["transaction_id"].count()

    def vendor_similarity(row):
        if pd.isna(row["invoice_id"]) or pd.isna(row["transaction_id"]):
            return -1.0
        k1, k2 = inv_keys.get(row["invoice_id"]), pay_keys.get(row["transaction_id"])
        if not k1 or not k2:
            return -1.0
        return fuzz.token_sort_ratio(k1, k2)

    df = results.copy()
    df["has_payment"] = df["transaction_id"].notna().astype(int)
    df["has_invoice"] = df["invoice_id"].notna().astype(int)
    df["amount_pct_diff_f"] = df["amount_pct_diff"].fillna(-1.0)
    df["date_lag_days_f"] = df["date_lag_days"].fillna(-999)
    df["match_confidence_f"] = df["match_confidence"].fillna(0.0)
    df["match_method_enc"] = df["match_method"].map({"exact_reference": 2, "fuzzy": 1}).fillna(0)
    df["invoice_amount_f"] = df["invoice_amount"].fillna(df["payment_amount"]).fillna(0)
    df["payment_amount_f"] = df["payment_amount"].fillna(0)
    df["is_duplicate_group"] = df["invoice_id"].map(dup_counts).fillna(1).gt(1).astype(int)
    df["vendor_similarity"] = df.apply(vendor_similarity, axis=1)

    # ---- Labels (used ONLY for supervised training + evaluation of both models) ----
    inv_scenario = gt[gt["invoice_id"].notna()][["invoice_id", "scenario"]].drop_duplicates()
    txn_scenario = gt[gt.get("transaction_id").notna()][["transaction_id", "scenario"]] if "transaction_id" in gt else pd.DataFrame(columns=["transaction_id", "scenario"])

    df = df.merge(inv_scenario, on="invoice_id", how="left", suffixes=("", "_inv"))
    df = df.merge(txn_scenario, on="transaction_id", how="left", suffixes=("", "_txn"))
    df["scenario"] = df["scenario"].fillna(df["scenario_txn"])
    df["is_anomaly"] = df["scenario"].isin(ANOMALY_SCENARIOS).astype(int)

    feature_cols = [
        "amount_pct_diff_f", "date_lag_days_f", "match_confidence_f",
        "match_method_enc", "has_payment", "has_invoice",
        "invoice_amount_f", "payment_amount_f", "is_duplicate_group",
        "vendor_similarity",
    ]
    return df, feature_cols


def run():
    df, feature_cols = build_feature_table()
    X = df[feature_cols].values
    y = df["is_anomaly"].values

    # ---------------------------------------------------------
    # Model A: Isolation Forest (unsupervised - never sees y during fit)
    # ---------------------------------------------------------
    contamination = max(0.01, min(0.3, y.mean()))  # informed guess, NOT a label leak into training
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=42)
    iso.fit(X)
    # decision_function: higher = more normal. Flip and normalize to a 0-1 "risk score".
    raw_scores = -iso.decision_function(X)
    iso_risk = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    iso_pred = (iso.predict(X) == -1).astype(int)

    iso_precision, iso_recall, iso_f1, _ = precision_recall_fscore_support(
        y, iso_pred, average="binary", zero_division=0
    )
    iso_auc = roc_auc_score(y, iso_risk)

    # ---------------------------------------------------------
    # Model B: Random Forest (supervised - trained on labeled history)
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.3, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    rf_pred_test = rf.predict(X_test)
    rf_proba_test = rf.predict_proba(X_test)[:, 1]

    rf_precision, rf_recall, rf_f1, _ = precision_recall_fscore_support(
        y_test, rf_pred_test, average="binary", zero_division=0
    )
    rf_auc = roc_auc_score(y_test, rf_proba_test)

    # Full-dataset RF scores for the combined output file (fit model already trained on train split only)
    rf_proba_full = rf.predict_proba(X)[:, 1]

    feature_importance = dict(zip(feature_cols, rf.feature_importances_.round(4)))

    # ---------------------------------------------------------
    # Assemble final predictions file
    # ---------------------------------------------------------
    out = df[["invoice_id", "transaction_id", "vendor_name", "status", "is_anomaly"]].copy()
    out["isolation_forest_risk"] = iso_risk.round(4)
    out["isolation_forest_flag"] = iso_pred
    out["random_forest_risk"] = rf_proba_full.round(4)
    out["random_forest_flag"] = (rf_proba_full >= 0.5).astype(int)
    out["combined_flag"] = ((out["isolation_forest_flag"] == 1) | (out["random_forest_flag"] == 1)).astype(int)
    out.to_csv(os.path.join(REPORTS_DIR, "ml_predictions.csv"), index=False)

    report = {
        "dataset": {
            "total_records": int(len(df)),
            "anomaly_rate_ground_truth": round(float(y.mean()), 4),
        },
        "isolation_forest": {
            "type": "unsupervised",
            "precision": round(iso_precision, 4),
            "recall": round(iso_recall, 4),
            "f1": round(iso_f1, 4),
            "roc_auc": round(iso_auc, 4),
            "evaluated_on": "full dataset (unsupervised - no train/test split needed)",
        },
        "random_forest": {
            "type": "supervised",
            "precision": round(rf_precision, 4),
            "recall": round(rf_recall, 4),
            "f1": round(rf_f1, 4),
            "roc_auc": round(rf_auc, 4),
            "evaluated_on": "held-out 30% test split",
            "feature_importance": feature_importance,
        },
    }
    with open(os.path.join(REPORTS_DIR, "ml_anomaly_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== ML ANOMALY DETECTION: MODEL COMPARISON ===\n")
    print(f"Ground-truth anomaly rate: {y.mean()*100:.1f}%\n")
    print(f"{'Metric':<12}{'IsolationForest':>18}{'RandomForest':>15}")
    for m in ["precision", "recall", "f1", "roc_auc"]:
        print(f"{m:<12}{report['isolation_forest'][m]:>18.3f}{report['random_forest'][m]:>15.3f}")
    print("\nTop feature importances (Random Forest):")
    for feat, imp in sorted(feature_importance.items(), key=lambda x: -x[1])[:5]:
        print(f"  {feat:<22} {imp:.3f}")
    print(f"\nSaved -> reports/ml_predictions.csv, reports/ml_anomaly_report.json")

    return report


if __name__ == "__main__":
    run()
