"""
Run the full reconciliation pipeline end to end, one stage at a time:
  1. Generate synthetic invoices/payments
  2. Ingest + clean + validate
  3. Reconcile (exact -> fuzzy matching)
  4. ML anomaly detection (Isolation Forest vs Random Forest)
  5. Reporting (Power BI export, KPIs, charts)

Usage:
    python3 run_pipeline.py
"""

import subprocess
import sys
import time

STAGES = [
    ("Generating synthetic dataset", "generate_data.py"),
    ("Ingesting & validating data", "ingest.py"),
    ("Running reconciliation engine", "reconcile.py"),
    ("Running ML anomaly detection", "ml_anomaly.py"),
    ("Building reports & Power BI export", "report.py"),
]


def main():
    print("=" * 60)
    print("AP RECONCILIATION PIPELINE")
    print("=" * 60)

    for label, script in STAGES:
        print(f"\n--- {label} ({script}) ---")
        start = time.time()
        result = subprocess.run([sys.executable, f"src/{script}"])
        elapsed = time.time() - start
        if result.returncode != 0:
            print(f"\nStage failed: {script}. Stopping pipeline.")
            sys.exit(1)
        print(f"({elapsed:.1f}s)")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE. See the reports/ directory for all outputs:")
    print("  - reports/master_reconciliation_export.csv  (Power BI data source)")
    print("  - reports/SUMMARY.md                         (human-readable report)")
    print("  - reports/charts/*.png                       (visuals)")
    print("=" * 60)


if __name__ == "__main__":
    main()
