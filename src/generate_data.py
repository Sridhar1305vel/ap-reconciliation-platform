"""
Synthetic Accounts-Payable Reconciliation Dataset Generator
------------------------------------------------------------
Generates two related tables:
  - invoices.csv : invoices issued by vendors, awaiting payment
  - payments.csv : payments recorded in the company's bank/ledger

Deliberately plants realistic discrepancy patterns so the
reconciliation engine and ML anomaly detector have real signal
to work with:
  1. clean_match      - invoice paid correctly, exact amount & vendor
  2. amount_mismatch  - payment differs slightly from invoice (short pay / overpay)
  3. duplicate_payment- same invoice paid twice
  4. orphan_payment   - payment with no corresponding invoice at all
  5. unpaid_invoice   - invoice with no payment yet (not necessarily an error)
  6. vendor_typo      - payment references a slightly misspelled vendor name
  7. date_anomaly     - payment posted implausibly before invoice date, or very late

Run:  python3 src/generate_data.py
Output: data/invoices.csv, data/payments.csv, data/ground_truth.csv
"""

import os
import random
import string
import uuid
from datetime import timedelta

import numpy as np
import pandas as pd
from faker import Faker

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)
fake = Faker()
Faker.seed(42)

N_INVOICES = 1200
CURRENCIES = ["GBP", "USD", "EUR"]

# ---------------------------------------------------------------
# 1. Vendor pool (with a few "typo twins" for fuzzy-matching tests)
# ---------------------------------------------------------------
vendors = [fake.unique.company() for _ in range(60)]


def typo_variant(name: str) -> str:
    """Return a slightly corrupted version of a vendor name."""
    name = list(name)
    op = random.choice(["swap", "drop", "case", "suffix"])
    if op == "swap" and len(name) > 3:
        i = random.randint(0, len(name) - 2)
        name[i], name[i + 1] = name[i + 1], name[i]
    elif op == "drop" and len(name) > 3:
        i = random.randint(0, len(name) - 1)
        del name[i]
    elif op == "case":
        name = [c.upper() if random.random() < 0.5 else c.lower() for c in name]
    elif op == "suffix":
        name = name + list(random.choice([" Ltd", " Inc", " LLP", ".", " "]))
    return "".join(name)


# ---------------------------------------------------------------
# 2. Generate invoices
# ---------------------------------------------------------------
invoices = []
for i in range(N_INVOICES):
    invoice_id = f"INV-{100000 + i}"
    vendor = random.choice(vendors)
    invoice_date = fake.date_between(start_date="-180d", end_date="-30d")
    due_date = invoice_date + timedelta(days=random.choice([15, 30, 45, 60]))
    amount = round(random.uniform(50, 25000), 2)
    currency = random.choices(CURRENCIES, weights=[0.6, 0.25, 0.15])[0]

    invoices.append(
        {
            "invoice_id": invoice_id,
            "vendor_name": vendor,
            "invoice_date": invoice_date,
            "due_date": due_date,
            "amount": amount,
            "currency": currency,
        }
    )

invoices_df = pd.DataFrame(invoices)

# ---------------------------------------------------------------
# 3. Generate payments, tagging each with a planted scenario label
#    (label kept ONLY in ground_truth.csv, never in the working files)
# ---------------------------------------------------------------
payments = []
ground_truth = []

scenario_weights = {
    "clean_match": 0.68,
    "amount_mismatch": 0.08,
    "duplicate_payment": 0.06,
    "unpaid_invoice": 0.08,
    "vendor_typo": 0.06,
    "date_anomaly": 0.04,
}
assert abs(sum(scenario_weights.values()) - 1.0) < 1e-9

scenarios = np.random.choice(
    list(scenario_weights.keys()),
    size=N_INVOICES,
    p=list(scenario_weights.values()),
)

txn_counter = 0


def new_txn_id():
    global txn_counter
    txn_counter += 1
    return f"TXN-{200000 + txn_counter}"


for inv, scenario in zip(invoices, scenarios):
    if scenario == "unpaid_invoice":
        ground_truth.append({"invoice_id": inv["invoice_id"], "scenario": scenario})
        continue  # no payment row at all

    base_payment = {
        "transaction_id": new_txn_id(),
        "vendor_name": inv["vendor_name"],
        "payment_date": inv["invoice_date"]
        + timedelta(days=random.randint(5, 40)),
        "amount": inv["amount"],
        "currency": inv["currency"],
        "reference_note": f"Payment for {inv['invoice_id']}",
    }

    if scenario == "clean_match":
        payments.append(base_payment)

    elif scenario == "amount_mismatch":
        p = dict(base_payment)
        delta = round(inv["amount"] * random.uniform(0.01, 0.08), 2)
        p["amount"] = round(inv["amount"] - delta if random.random() < 0.5 else inv["amount"] + delta, 2)
        payments.append(p)

    elif scenario == "duplicate_payment":
        payments.append(dict(base_payment))
        dup = dict(base_payment)
        dup["transaction_id"] = new_txn_id()
        dup["payment_date"] = base_payment["payment_date"] + timedelta(days=random.randint(1, 5))
        payments.append(dup)

    elif scenario == "vendor_typo":
        p = dict(base_payment)
        p["vendor_name"] = typo_variant(inv["vendor_name"])
        # Real-world bank narrations rarely repeat the invoice number when the
        # vendor name itself is garbled/abbreviated by the bank's system, so
        # most vendor-typo cases ALSO drop the parseable invoice reference -
        # this is what forces the reconciliation engine into fuzzy matching.
        if random.random() < 0.75:
            p["reference_note"] = random.choice(
                ["Vendor payment - see remittance advice", "BACS TRANSFER", "Supplier payment", "PAYMENT REF UNKNOWN"]
            )
        payments.append(p)

    elif scenario == "date_anomaly":
        p = dict(base_payment)
        # payment posted BEFORE the invoice was even issued, or very late
        if random.random() < 0.5:
            p["payment_date"] = inv["invoice_date"] - timedelta(days=random.randint(3, 15))
        else:
            p["payment_date"] = inv["invoice_date"] + timedelta(days=random.randint(120, 200))
        payments.append(p)

    ground_truth.append({"invoice_id": inv["invoice_id"], "scenario": scenario})

# ---------------------------------------------------------------
# 3b. Additionally strip the reference on a slice of otherwise-clean
#     payments, since real bank feeds frequently omit invoice numbers
#     even when nothing else is wrong. These should still be recoverable
#     via fuzzy (vendor + amount + date) matching.
# ---------------------------------------------------------------
UNREFERENCED_SAMPLE_RATE = 0.10
for p in payments:
    if p.get("reference_note", "").startswith("Payment for INV") and random.random() < UNREFERENCED_SAMPLE_RATE:
        p["reference_note"] = random.choice(
            ["Faster Payment received", "Supplier settlement", "TRANSFER", "Invoice settlement - ref n/a"]
        )

# ---------------------------------------------------------------
# 4. Sprinkle in pure orphan payments (no invoice at all — e.g. fraud/error)
# ---------------------------------------------------------------
N_ORPHANS = 40
for _ in range(N_ORPHANS):
    vendor = random.choice(vendors)
    fake_invoice_ref = f"INV-{random.randint(900000, 999999)}"  # doesn't exist
    p = {
        "transaction_id": new_txn_id(),
        "vendor_name": vendor,
        "payment_date": fake.date_between(start_date="-150d", end_date="-1d"),
        "amount": round(random.uniform(50, 25000), 2),
        "currency": random.choices(CURRENCIES, weights=[0.6, 0.25, 0.15])[0],
        "reference_note": f"Payment for {fake_invoice_ref}",
    }
    payments.append(p)
    ground_truth.append({"invoice_id": None, "transaction_id": p["transaction_id"], "scenario": "orphan_payment"})

payments_df = pd.DataFrame(payments).sample(frac=1, random_state=42).reset_index(drop=True)
ground_truth_df = pd.DataFrame(ground_truth)

# ---------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------
invoices_df.to_csv(os.path.join(DATA_DIR, "invoices.csv"), index=False)
payments_df.to_csv(os.path.join(DATA_DIR, "payments.csv"), index=False)
ground_truth_df.to_csv(os.path.join(DATA_DIR, "ground_truth.csv"), index=False)

print(f"Invoices:      {len(invoices_df)} rows -> data/invoices.csv")
print(f"Payments:      {len(payments_df)} rows -> data/payments.csv")
print(f"Ground truth:  {len(ground_truth_df)} rows -> data/ground_truth.csv")
print("\nScenario distribution:")
print(ground_truth_df["scenario"].value_counts())
