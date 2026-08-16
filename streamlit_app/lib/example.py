# =============================================================================
# lib/example.py -- keyless worked example for the close anomaly scan
# =============================================================================
# Recomputes detection from the committed sample data at default thresholds,
# then reads the captured AI triage verbatim from the committed JSON file.
# The pattern follows Project 1: deterministic parts recomputed, AI parts
# read verbatim, never fabricated.
# =============================================================================

import json
from pathlib import Path

from config import (
    CLOSE_FILE, HISTORY_FILE,
    GLOBAL_THRESHOLD, ACCOUNT_THRESHOLDS,
    MODIFIED_Z_CUTOFF, NEAR_ZERO,
)
from src.step1_data_loader import load_close, load_history
from src.step2_detection_engine import detect_anomalies
from streamlit_app.lib.triage_helpers import (
    triage_fingerprint, match_triage_to_flagged,
)

SAMPLE_TRIAGE = Path(__file__).resolve().parent.parent.parent / "data" / "sample_triage.json"


def load_example():
    """Load the keyless worked example.

    Returns a dict with flagged, fingerprint, narrative, triage,
    matched, missing, tokens_in, tokens_out, and stop_reason.
    """
    close_df = load_close(CLOSE_FILE)
    history = load_history(HISTORY_FILE)
    flagged, _ = detect_anomalies(
        close_df, history,
        account_thresholds=ACCOUNT_THRESHOLDS,
        global_threshold=GLOBAL_THRESHOLD,
        z_cutoff=MODIFIED_Z_CUTOFF,
        near_zero=NEAR_ZERO,
    )
    fp = triage_fingerprint(flagged)

    with open(SAMPLE_TRIAGE, encoding="utf-8") as f:
        captured = json.load(f)

    matched, missing = match_triage_to_flagged(captured["triage"], flagged)

    return {
        "flagged": flagged,
        "fingerprint": fp,
        "narrative": captured["narrative"],
        "triage": captured["triage"],
        "matched": matched,
        "missing": missing,
        "tokens_in": captured["tokens_in"],
        "tokens_out": captured["tokens_out"],
        "stop_reason": captured["stop_reason"],
    }
