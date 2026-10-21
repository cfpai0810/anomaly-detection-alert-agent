# =============================================================================
# lib/audit.py -- session audit for the close anomaly scan
# =============================================================================
# build_audit_record() is a pure function (no Streamlit dependency) so it can
# be tested in the main test suite.  record_run() and get_runs() are thin
# Streamlit wrappers that store the records in session_state. sign_run()
# adds a reviewer's sign-off to its run's row: who, when, and the flagged
# accounts shown at sign-off (SignOff.record from src/signoff.py).
# =============================================================================

from datetime import datetime, timezone

from streamlit_app.lib.cost import estimate_cost

AUDIT_KEY = "scan_audit"


def build_audit_record(flagged, eff_thresholds, close_hash, history_hash,
                       tok_in, tok_out, stop_reason):
    """Build one audit record from the run's inputs and outputs."""
    at, gt, zc, nz = eff_thresholds
    requires_review = (
        len(flagged) > 0
        or stop_reason == "max_tokens"
        or (tok_out < 150 and stop_reason != "max_tokens")
    )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_flagged": len(flagged),
        "flagged_accounts": [f["account"] for f in flagged],
        "effective_thresholds": {
            "account_thresholds": at,
            "global_threshold": gt,
            "z_cutoff": zc,
            "near_zero": nz,
        },
        "close_hash": close_hash,
        "history_hash": history_hash,
        "tokens_in": tok_in,
        "tokens_out": tok_out,
        "cost": estimate_cost(tok_in, tok_out),
        "requires_review": requires_review,
        "stop_reason": stop_reason,
    }


def record_run(flagged, eff_thresholds, close_hash, history_hash,
               tok_in, tok_out, stop_reason):
    """Record one triage run in the Streamlit session audit trail."""
    import streamlit as st
    record = build_audit_record(
        flagged, eff_thresholds, close_hash, history_hash,
        tok_in, tok_out, stop_reason)
    if AUDIT_KEY not in st.session_state:
        st.session_state[AUDIT_KEY] = []
    st.session_state[AUDIT_KEY].insert(0, record)
    return record


def sign_run(run, signed, acknowledged):
    """Add a sign-off to the row of run number `run`, as the session audit
    list numbers its runs (the first run is 1). signed is the SignOff from
    src/signoff.py; acknowledged is the list of flagged accounts shown at
    sign-off. A later sign-off of the same run replaces it."""
    import streamlit as st
    runs = st.session_state[AUDIT_KEY]
    record = runs[len(runs) - run]
    record.update(signed.record(acknowledged))
    return record


def get_runs():
    """Return the session audit trail (newest first), or an empty list."""
    import streamlit as st
    return st.session_state.get(AUDIT_KEY, [])
