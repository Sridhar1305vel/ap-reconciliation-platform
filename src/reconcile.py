"""
Reconciliation Engine
----------------------
Two-stage matching, mirroring how real AP reconciliation tools work:

  STAGE 1 - EXACT MATCH
    Match payments to invoices via the invoice_id parsed out of the
    payment's free-text reference_note. Fast, high-confidence, but only
    works when the reference field was filled in correctly.

  STAGE 2 - FUZZY MATCH (for anything Stage 1 couldn't resolve)
    For invoices with no exact match and payments with no exact match,
    score every remaining (invoice, payment) pair on:
      - vendor name similarity (rapidfuzz token_sort_ratio)
      - amount closeness (% difference)
      - date plausibility (payment falls in a sane window vs invoice date)
    and greedily accept the best mutual matches above a confidence floor.
    This is what catches typo'd vendor names that Stage 1 would miss.

Every invoice/payment ends in exactly one of these statuses:
  - matched_clean          : exact match, amount identical (within cents)
  - matched_amount_mismatch: exact/fuzzy match, amount differs
  - matched_duplicate      : invoice has 2+ payments referencing it
  - matched_fuzzy          : resolved only via Stage 2, tagged with confidence
  - date_anomaly           : matched, but payment date is implausible
  - unmatched_invoice      : no payment found (possibly just not yet paid)
  - unmatched_payment      : no invoice found at all (orphan payment)

Output: reports/reconciliation_results.csv
"""

import os

import pandas as pd
from rapidfuzz import fuzz

from ingest import load_and_clean_invoices, load_and_clean_payments

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

AMOUNT_TOLERANCE_PCT = 0.005      # 0.5% -> treated as "clean" (rounding noise)
FUZZY_VENDOR_FLOOR = 70           # min vendor similarity score (0-100) to even consider a fuzzy pair
FUZZY_AMOUNT_TOLERANCE_PCT = 0.10 # fuzzy candidates must be within 10% on amount
FUZZY_CONFIDENCE_FLOOR = 0.55     # combined score (0-1) needed to accept a fuzzy match
DATE_MIN_LAG_DAYS = 0             # payment shouldn't predate invoice
DATE_MAX_LAG_DAYS = 100           # payment this late vs invoice is flagged as anomalous


def _amount_pct_diff(a, b):
    if a == 0:
        return 1.0 if b != 0 else 0.0
    return abs(a - b) / abs(a)


def _date_status(invoice_date, payment_date):
    lag = (payment_date - invoice_date).days
    if lag < DATE_MIN_LAG_DAYS or lag > DATE_MAX_LAG_DAYS:
        return "date_anomaly", lag
    return "ok", lag


def reconcile(invoices: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    invoices = invoices.copy()
    payments = payments.copy()

    results = []
    matched_payment_ids = set()

    # ---------------------------------------------------------
    # STAGE 1: exact match via parsed reference_note
    # ---------------------------------------------------------
    valid_invoice_ids = set(invoices["invoice_id"])

    for _, inv in invoices.iterrows():
        candidates = payments[
            (payments["referenced_invoice_id"] == inv["invoice_id"])
        ]

        if len(candidates) == 0:
            continue  # goes to unmatched pool, handled after Stage 2

        if len(candidates) > 1:
            # Duplicate payment scenario: flag ALL of them
            for _, pay in candidates.iterrows():
                pct_diff = _amount_pct_diff(inv["amount"], pay["amount"])
                date_stat, lag = _date_status(inv["invoice_date"], pay["payment_date"])
                results.append({
                    "invoice_id": inv["invoice_id"],
                    "transaction_id": pay["transaction_id"],
                    "vendor_name": inv["vendor_name"],
                    "invoice_amount": inv["amount"],
                    "payment_amount": pay["amount"],
                    "amount_pct_diff": round(pct_diff, 4),
                    "date_lag_days": lag,
                    "match_method": "exact_reference",
                    "match_confidence": 1.0,
                    "status": "matched_duplicate" if date_stat == "ok" else "matched_duplicate+date_anomaly",
                })
                matched_payment_ids.add(pay["transaction_id"])
            continue

        # exactly one candidate
        pay = candidates.iloc[0]
        pct_diff = _amount_pct_diff(inv["amount"], pay["amount"])
        date_stat, lag = _date_status(inv["invoice_date"], pay["payment_date"])

        if date_stat == "date_anomaly":
            status = "date_anomaly"
        elif pct_diff > AMOUNT_TOLERANCE_PCT:
            status = "matched_amount_mismatch"
        else:
            status = "matched_clean"

        results.append({
            "invoice_id": inv["invoice_id"],
            "transaction_id": pay["transaction_id"],
            "vendor_name": inv["vendor_name"],
            "invoice_amount": inv["amount"],
            "payment_amount": pay["amount"],
            "amount_pct_diff": round(pct_diff, 4),
            "date_lag_days": lag,
            "match_method": "exact_reference",
            "match_confidence": 1.0,
            "status": status,
        })
        matched_payment_ids.add(pay["transaction_id"])

    matched_invoice_ids = {r["invoice_id"] for r in results}

    # ---------------------------------------------------------
    # STAGE 2: fuzzy match remaining invoices <-> remaining payments
    # ---------------------------------------------------------
    remaining_invoices = invoices[~invoices["invoice_id"].isin(matched_invoice_ids)].copy()
    remaining_payments = payments[~payments["transaction_id"].isin(matched_payment_ids)].copy()

    fuzzy_pairs = []
    for _, inv in remaining_invoices.iterrows():
        for _, pay in remaining_payments.iterrows():
            if inv["currency"] != pay["currency"]:
                continue
            amt_diff = _amount_pct_diff(inv["amount"], pay["amount"])
            if amt_diff > FUZZY_AMOUNT_TOLERANCE_PCT:
                continue
            vendor_score = fuzz.token_sort_ratio(inv["vendor_key"], pay["vendor_key"])
            if vendor_score < FUZZY_VENDOR_FLOOR:
                continue

            # Combined confidence: weighted blend of vendor similarity + amount closeness
            amount_score = max(0.0, 1 - (amt_diff / FUZZY_AMOUNT_TOLERANCE_PCT))
            confidence = 0.6 * (vendor_score / 100) + 0.4 * amount_score

            if confidence >= FUZZY_CONFIDENCE_FLOOR:
                fuzzy_pairs.append((confidence, inv["invoice_id"], pay["transaction_id"]))

    # Greedy assignment: accept highest-confidence pairs first, each invoice/payment used once
    fuzzy_pairs.sort(key=lambda x: x[0], reverse=True)
    used_invoices, used_payments = set(), set()

    for confidence, inv_id, txn_id in fuzzy_pairs:
        if inv_id in used_invoices or txn_id in used_payments:
            continue
        used_invoices.add(inv_id)
        used_payments.add(txn_id)

        inv = invoices.loc[invoices["invoice_id"] == inv_id].iloc[0]
        pay = payments.loc[payments["transaction_id"] == txn_id].iloc[0]
        pct_diff = _amount_pct_diff(inv["amount"], pay["amount"])
        date_stat, lag = _date_status(inv["invoice_date"], pay["payment_date"])

        status = "date_anomaly" if date_stat == "date_anomaly" else (
            "matched_amount_mismatch" if pct_diff > AMOUNT_TOLERANCE_PCT else "matched_fuzzy"
        )

        results.append({
            "invoice_id": inv_id,
            "transaction_id": txn_id,
            "vendor_name": inv["vendor_name"],
            "invoice_amount": inv["amount"],
            "payment_amount": pay["amount"],
            "amount_pct_diff": round(pct_diff, 4),
            "date_lag_days": lag,
            "match_method": "fuzzy",
            "match_confidence": round(confidence, 3),
            "status": status,
        })
        matched_invoice_ids.add(inv_id)
        matched_payment_ids.add(txn_id)

    # ---------------------------------------------------------
    # Leftovers: truly unmatched invoices and orphan payments
    # ---------------------------------------------------------
    for _, inv in invoices[~invoices["invoice_id"].isin(matched_invoice_ids)].iterrows():
        results.append({
            "invoice_id": inv["invoice_id"],
            "transaction_id": None,
            "vendor_name": inv["vendor_name"],
            "invoice_amount": inv["amount"],
            "payment_amount": None,
            "amount_pct_diff": None,
            "date_lag_days": None,
            "match_method": None,
            "match_confidence": 0.0,
            "status": "unmatched_invoice",
        })

    for _, pay in payments[~payments["transaction_id"].isin(matched_payment_ids)].iterrows():
        results.append({
            "invoice_id": None,
            "transaction_id": pay["transaction_id"],
            "vendor_name": pay["vendor_name"],
            "invoice_amount": None,
            "payment_amount": pay["amount"],
            "amount_pct_diff": None,
            "date_lag_days": None,
            "match_method": None,
            "match_confidence": 0.0,
            "status": "unmatched_payment",
        })

    return pd.DataFrame(results)


def run():
    invoices, _ = load_and_clean_invoices()
    payments, _ = load_and_clean_payments()

    results = reconcile(invoices, payments)
    out_path = os.path.join(REPORTS_DIR, "reconciliation_results.csv")
    results.to_csv(out_path, index=False)

    print(f"Saved reconciliation results -> {out_path}\n")
    print("=== RECONCILIATION SUMMARY ===")
    print(results["status"].value_counts().to_string())
    print(f"\nTotal rows: {len(results)}")
    return results


if __name__ == "__main__":
    run()
