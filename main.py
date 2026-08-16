# =============================================================================
# main.py — Anomaly Detection and Alert Agent
# Pass 1: flat script, console output, understand every line
# =============================================================================
#
# Python DETECTS the statistical anomalies. Claude TRIAGES them: ranks,
# judges timing versus permanent, and hypothesises a cause. The agent
# never suppresses a flag; every Python-flagged account stays on the list.
# =============================================================================

from dotenv import load_dotenv

load_dotenv()

from src.step1_data_loader    import load_close, load_history
from src.step2_detection_engine import detect_anomalies
from src.step3_triage_agent     import (
    build_prompt, call_claude, parse_triage_json, strip_json_block,
)
from src.step4_output_writer    import write_output, write_pdf, export_anomalies_csv
from config import (
    CLOSE_FILE, HISTORY_FILE,
    DEFAULT_ENTITY, CLOSE_PERIOD,
)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    close_df = load_close(CLOSE_FILE)
    history  = load_history(HISTORY_FILE)

    flagged, flags = detect_anomalies(close_df, history)

    print("\n[OK] Detection complete")
    print("     Accounts scanned: {}".format(len(close_df)))
    print("     Flagged:          {}".format(len(flagged)))
    print("     Flags raised:     {}".format(len(flags)))
    for f in flagged:
        reasons = []
        if f["material_benchmarks"]:
            reasons.append("material vs " + "+".join(f["material_benchmarks"]))
        if f["stable_moved"]:
            reasons.append("stable account moved")
        print("     --> {}: {}".format(f["account"], "; ".join(reasons)))

    if not flagged:
        print("\n[DONE] No anomalies flagged. Clean close.")
    else:
        system_prompt, user_prompt = build_prompt(flagged, DEFAULT_ENTITY, CLOSE_PERIOD)

        try:
            raw_response, tok_in, tok_out, stop_reason = call_claude(system_prompt, user_prompt)
        except RuntimeError as exc:
            print("\n[ERROR] {}".format(exc))
            raise SystemExit(1)

        # Split the response: narrative for humans, JSON for the CSV artefact
        triage_narrative = strip_json_block(raw_response)
        triage_json      = parse_triage_json(raw_response)

        print("\n" + "=" * 64)
        print("CLOSE ANOMALY TRIAGE — {}".format(DEFAULT_ENTITY))
        print("=" * 64)
        print(triage_narrative)
        print("=" * 64)
        if triage_json is None:
            print("[WARN] Structured JSON not parsed. CSV will use detection data only.")
        else:
            print("[OK] Parsed structured triage for {} accounts.".format(len(triage_json)))

        out_path, audit = write_output(
            triage_narrative, flagged, flags, tok_in, tok_out, stop_reason
        )

        pdf_path = write_pdf(
            triage_narrative, flagged, triage_json, tok_in, tok_out
        )

        csv_path = export_anomalies_csv(flagged, triage_json)

        print("\n[DONE] Triage complete.")
        print("       Output: {}".format(out_path.name))
        if audit["requires_review"]:
            print("\n" + "!" * 64)
            print("  HUMAN REVIEW REQUIRED — {} account(s) flagged".format(len(flagged)))
            print("!" * 64)
