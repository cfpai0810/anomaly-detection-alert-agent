# =============================================================================
# lib/triage_helpers.py — pure helpers for the AI triage section
# =============================================================================
# No Streamlit dependency. Tested in the main test suite.
# =============================================================================

import json
import html as _html

from src.step4_output_writer import _normalise_account


PRIORITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def triage_fingerprint(flagged):
    """Deterministic string from the flagged accounts and their
    triage-relevant detail, used to detect when the triage cards are stale.

    The fingerprint captures what the model was actually shown: which
    accounts, which benchmarks were material (and why), and the volatility
    label. Raw variance numbers are data-derived and do not change with
    thresholds, so they are omitted. The triage stays valid as long as
    re-running would send the model the same prompt about the same accounts.
    """
    entries = []
    for f in flagged:
        benchmarks = {}
        for bench in sorted(f["comparisons"]):
            c = f["comparisons"][bench]
            benchmarks[bench] = {
                "material": c["material"],
                "reason": c.get("reason"),
            }
        entries.append({
            "account": f["account"],
            "benchmarks": benchmarks,
            "volatility": f["volatility"],
        })
    return json.dumps(entries, sort_keys=True)


def match_triage_to_flagged(triage_json, flagged):
    """Match triage entries to flagged accounts by normalised name.

    Returns (matched, missing) where matched is a dict mapping the original
    flagged account name to its triage entry, and missing is the list of
    flagged account names the model did not return.
    """
    triage_lookup = {}
    for t in (triage_json or []):
        triage_lookup[_normalise_account(t["account"])] = t

    matched = {}
    missing = []
    for f in flagged:
        key = _normalise_account(f["account"])
        if key in triage_lookup:
            matched[f["account"]] = triage_lookup[key]
        else:
            missing.append(f["account"])

    return matched, missing


def sort_by_priority(entries):
    """Sort triage entries HIGH first, then MEDIUM, then LOW."""
    return sorted(
        entries,
        key=lambda t: PRIORITY_ORDER.get(
            t.get("priority", "LOW").upper(), 3))


def escape_text(text):
    """Escape text for safe inclusion in an HTML context."""
    return _html.escape(str(text))
