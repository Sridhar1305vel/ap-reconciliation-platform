"""
Unit tests for the reconciliation pipeline's core logic.
Run with: pytest tests/ -v
"""

import os
import sys
from datetime import datetime

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from ingest import _normalize_vendor
from reconcile import _amount_pct_diff, _date_status, reconcile


# ---------------------------------------------------------------
# Vendor normalization
# ---------------------------------------------------------------
def test_normalize_vendor_case_insensitive():
    assert _normalize_vendor("WiLLIamS And soNs") == _normalize_vendor("Williams and Sons")


def test_normalize_vendor_strips_legal_suffixes():
    assert _normalize_vendor("Acme Ltd") == _normalize_vendor("Acme")


def test_normalize_vendor_handles_nan():
    assert _normalize_vendor(float("nan")) == ""


# ---------------------------------------------------------------
# Amount comparison
# ---------------------------------------------------------------
def test_amount_pct_diff_identical():
    assert _amount_pct_diff(100.0, 100.0) == 0.0


def test_amount_pct_diff_nonzero():
    assert _amount_pct_diff(100.0, 110.0) == pytest.approx(0.10)


def test_amount_pct_diff_zero_base_nonzero_other():
    assert _amount_pct_diff(0.0, 50.0) == 1.0


# ---------------------------------------------------------------
# Date plausibility
# ---------------------------------------------------------------
def test_date_status_ok_within_window():
    status, lag = _date_status(datetime(2026, 1, 1), datetime(2026, 1, 20))
    assert status == "ok"
    assert lag == 19


def test_date_status_flags_payment_before_invoice():
    status, lag = _date_status(datetime(2026, 1, 20), datetime(2026, 1, 1))
    assert status == "date_anomaly"
    assert lag < 0


def test_date_status_flags_very_late_payment():
    status, lag = _date_status(datetime(2026, 1, 1), datetime(2026, 6, 1))
    assert status == "date_anomaly"


# ---------------------------------------------------------------
# Reconciliation engine, end to end on a tiny fixture
# ---------------------------------------------------------------
@pytest.fixture
def tiny_fixture():
    invoices = pd.DataFrame([
        {"invoice_id": "INV-1", "vendor_name": "Acme Ltd", "vendor_key": "acme",
         "invoice_date": pd.Timestamp("2026-01-01"), "due_date": pd.Timestamp("2026-01-31"),
         "amount": 1000.0, "currency": "GBP"},
        {"invoice_id": "INV-2", "vendor_name": "Beta Co", "vendor_key": "beta co",
         "invoice_date": pd.Timestamp("2026-01-05"), "due_date": pd.Timestamp("2026-02-04"),
         "amount": 500.0, "currency": "GBP"},
    ])
    payments = pd.DataFrame([
        {"transaction_id": "TXN-1", "vendor_name": "Acme Ltd", "vendor_key": "acme",
         "payment_date": pd.Timestamp("2026-01-10"), "amount": 1000.0, "currency": "GBP",
         "reference_note": "Payment for INV-1", "referenced_invoice_id": "INV-1"},
        # Beta Co payment has NO parseable reference -> must be recovered via fuzzy matching
        {"transaction_id": "TXN-2", "vendor_name": "Beta Co", "vendor_key": "beta co",
         "payment_date": pd.Timestamp("2026-01-12"), "amount": 500.0, "currency": "GBP",
         "reference_note": "BACS TRANSFER", "referenced_invoice_id": None},
    ])
    return invoices, payments


def test_reconcile_exact_match(tiny_fixture):
    invoices, payments = tiny_fixture
    results = reconcile(invoices, payments)
    row = results[results["invoice_id"] == "INV-1"].iloc[0]
    assert row["status"] == "matched_clean"
    assert row["match_method"] == "exact_reference"


def test_reconcile_recovers_unreferenced_payment_via_fuzzy(tiny_fixture):
    invoices, payments = tiny_fixture
    results = reconcile(invoices, payments)
    row = results[results["invoice_id"] == "INV-2"].iloc[0]
    assert row["transaction_id"] == "TXN-2"
    assert row["match_method"] == "fuzzy"


def test_reconcile_flags_true_orphan():
    invoices = pd.DataFrame([
        {"invoice_id": "INV-1", "vendor_name": "Acme Ltd", "vendor_key": "acme",
         "invoice_date": pd.Timestamp("2026-01-01"), "due_date": pd.Timestamp("2026-01-31"),
         "amount": 1000.0, "currency": "GBP"},
    ])
    payments = pd.DataFrame([
        {"transaction_id": "TXN-99", "vendor_name": "Totally Unrelated Inc", "vendor_key": "totally unrelated",
         "payment_date": pd.Timestamp("2026-01-10"), "amount": 9999.0, "currency": "GBP",
         "reference_note": "mystery payment", "referenced_invoice_id": None},
    ])
    results = reconcile(invoices, payments)
    orphan_row = results[results["transaction_id"] == "TXN-99"].iloc[0]
    assert orphan_row["status"] == "unmatched_payment"
    unpaid_row = results[results["invoice_id"] == "INV-1"].iloc[0]
    assert unpaid_row["status"] == "unmatched_invoice"
