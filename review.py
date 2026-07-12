# =============================================================================
# review.py — Controller sign-off for the close anomaly review
# =============================================================================
# Usage:
#   python review.py
#   python review.py "Chun-Feng Pai"
#
# When anomalies are flagged, the audit log sets requires_review = true.
# The controller reviews the PDF, fills in the disposition column of the
# anomalies CSV (accept / dismiss / investigate) for each flagged account,
# then runs this to record the sign-off. It records who reviewed and when,
# and checks that every anomaly has a disposition, warning if any are blank.
# =============================================================================

import sys
import json
from datetime import datetime, timezone

import pandas as pd

from config import AUDIT_LOG


def _disposition_status(csv_file):
    """Count how many anomalies have a disposition filled in. Returns
    (filled, total) or (None, None) if the CSV cannot be read."""
    if not csv_file:
        return None, None
    try:
        df = pd.read_csv(csv_file)
    except (FileNotFoundError, OSError):
        return None, None
    if "disposition" not in df.columns:
        return None, None
    total  = len(df)
    filled = int(df["disposition"].notna().sum())
    # Treat empty strings as unfilled too
    if df["disposition"].dtype == object:
        filled = int((df["disposition"].fillna("").astype(str).str.strip() != "").sum())
    return filled, total


def mark_reviewed(reviewer):
    if not AUDIT_LOG.exists():
        print("No audit log found at {}".format(AUDIT_LOG))
        return

    lines = AUDIT_LOG.read_text(encoding="utf-8").strip().split("\n")
    if not lines or not lines[-1].strip():
        print("Audit log is empty.")
        return

    last = json.loads(lines[-1])

    if last.get("human_reviewed"):
        print("Latest run ({}) was already reviewed by {} at {}.".format(
            last.get("run_id", "?"),
            last.get("reviewed_by", "?"),
            last.get("reviewed_at", "?"),
        ))
        return

    # Check disposition completeness in the linked CSV
    filled, total = _disposition_status(last.get("csv_file"))
    if total is not None:
        if filled < total:
            print("[WARN] {} of {} anomalies have a disposition. {} still blank.".format(
                filled, total, total - filled))
            print("       Fill in the disposition column of:")
            from pathlib import Path
            print("       {}".format(Path(last.get("csv_file", "")).name))
            confirm = input("Sign off anyway? (y/N): ").strip().lower()
            if confirm != "y":
                print("Sign-off cancelled. No changes made.")
                return
        else:
            print("[OK] All {} anomalies have a disposition.".format(total))

    last["human_reviewed"]        = True
    last["reviewed_by"]           = reviewer
    last["reviewed_at"]           = datetime.now(timezone.utc).isoformat()
    if total is not None:
        last["dispositions_filled"] = filled
        last["dispositions_total"]  = total
    lines[-1] = json.dumps(last)
    AUDIT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Run {} signed off by {}.".format(last["run_id"], reviewer))
    if last.get("flagged_accounts"):
        print("Anomalies reviewed:")
        for a in last["flagged_accounts"]:
            print("  - {}".format(a))


if __name__ == "__main__":
    reviewer = sys.argv[1] if len(sys.argv) > 1 else "Controller"
    mark_reviewed(reviewer)
