# =============================================================================
# views/close_scan.py: the close anomaly scan surface
# =============================================================================
# Phase 6: sensitivity sliders drive detection live; AI triage runs with a
# key or shows a keyless worked example at default thresholds. Each triage
# run is recorded in a session audit trail. A live triage ends with the
# Sign-off section (the rule in src/signoff.py, after Project 6): its PDF is
# a DRAFT until a named reviewer signs it.
# =============================================================================

import hashlib
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit_app.lib.theme import (
    inject_css, header_html, priority_badge_html, status_chip_html,
    MUTED, FLAG_RED, AMBER, GREEN,
)
from streamlit_app.lib.key_gate import render_key_gate
from streamlit_app.lib.sensitivity import (
    LEVELS, BASELINE_INDEX,
    effective_thresholds,
)
from streamlit_app.lib.triage_helpers import (
    triage_fingerprint, match_triage_to_flagged,
    sort_by_priority, escape_text, flag_reasons, outstanding_lines,
)
from streamlit_app.lib.cost import format_cost
from streamlit_app.lib.errors import friendly_message
from streamlit_app.lib.audit import record_run, get_runs, sign_run
from streamlit_app.lib.example import load_example
from src.step1_data_loader import load_close, load_history
from src.step2_detection_engine import detect_anomalies
from src.step3_triage_agent import (
    build_prompt, call_claude, parse_triage_json, strip_json_block,
)
from src.step4_output_writer import (
    build_pdf_bytes, build_csv_bytes, file_sha256, format_money,
)
from src.signoff import (
    FINAL, SCOPE_NOTE, STALE_CAPTION, WORD_HELP, WORD_LABEL,
    SignOff, pdf_status_line, sign_off,
)
from config import (
    CLOSE_FILE, HISTORY_FILE,
    DEFAULT_ENTITY, CLOSE_PERIOD,
    CURRENCY_SYMBOL, BENCHMARKS,
    ACCOUNT_THRESHOLDS,
    GLOBAL_THRESHOLD, MODIFIED_Z_CUTOFF, NEAR_ZERO,
)


# -- CSS ----------------------------------------------------------------------
st.markdown(inject_css(), unsafe_allow_html=True)

# -- Header -------------------------------------------------------------------
st.markdown(header_html(
    "Close scan",
    "Scan the month-end close for statistical anomalies, adjust the "
    "detection sensitivity, and triage flagged accounts with the AI agent."),
    unsafe_allow_html=True)
st.caption("{}, Close {}".format(DEFAULT_ENTITY, CLOSE_PERIOD))

# -- Key gate (sidebar) ------------------------------------------------------
client = render_key_gate()
st.session_state["live_client"] = client


# =============================================================================
# DATA LOAD
# =============================================================================

@st.cache_data
def _load_close():
    return load_close(CLOSE_FILE)

@st.cache_data
def _load_history():
    return load_history(HISTORY_FILE)

close_df = _load_close()
history  = _load_history()


# =============================================================================
# THE TABS: the worked example first, then everything a live run uses
# =============================================================================

# The worked example is the first tab, so every way onto this page (Home's
# button, the sidebar, an address) opens on it. Everything a live run uses
# (the close, the sensitivity controls, the scan results and the triage)
# sits in Run it yourself, so the tab bar is directly under the banner and
# nothing above it changes when those controls do. The fixed key keeps the
# chosen tab when a rerun changes the page around it.
tab_example, tab_live = st.tabs(
    ["View a worked example", "Run it yourself"], key="cs_triage_tabs")


with tab_live:
    # =========================================================================
    # THE CLOSE TABLE
    # =========================================================================

    st.markdown("#### The close")
    st.caption(
        "The month-end close: each account with its actual and three benchmarks. "
        "This is the input the detection engine scans.")

    display_df = close_df[["account", "actual", "prior", "budget", "forecast"]].copy()
    for col in ["actual", "prior", "budget", "forecast"]:
        display_df[col] = display_df[col].apply(format_money)
    display_df.columns = ["Account", "Actual", "Prior month", "Budget", "Forecast"]

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.caption("{} accounts, {} trailing months of history each.".format(
        len(close_df), len(next(iter(history.values())))))

    st.markdown("---")


    # =========================================================================
    # DETECTION SENSITIVITY: sliders or custom thresholds
    # =========================================================================

    st.markdown("#### Detection sensitivity")

    threshold_mode = st.radio(
        "Choose how to set detection thresholds",
        ["Sensitivity sliders", "Custom thresholds"],
        horizontal=True,
        key="threshold_mode",
        label_visibility="collapsed",
    )

    if threshold_mode == "Sensitivity sliders":
        st.caption(
            "Higher sensitivity flags more accounts. Medium matches the default "
            "thresholds used by the CLI tool.")

        col_mat, col_vol = st.columns(2)

        with col_mat:
            mat_level = st.select_slider(
                "Materiality sensitivity",
                options=LEVELS,
                value=LEVELS[BASELINE_INDEX],
                help="How large a variance must be to flag. Higher sensitivity "
                     "means smaller thresholds, so smaller variances are caught.",
                key="mat_level",
            )

        with col_vol:
            vol_level = st.select_slider(
                "Volatility sensitivity",
                options=LEVELS,
                value=LEVELS[BASELINE_INDEX],
                help="How unusual a value must be relative to the account's own "
                     "history. Higher sensitivity catches smaller swings.",
                key="vol_level",
            )

        eff_at, eff_gt, eff_zc, eff_nz = effective_thresholds(
            mat_level, vol_level)

        _MAT_DESCRIPTIONS = {
            "Very low": "only very large variances are flagged",
            "Low": "only clearly significant variances are flagged",
            "Medium": "flags variances that most finance teams would "
                      "investigate",
            "High": "catches smaller variances that might otherwise be "
                    "overlooked",
            "Very high": "flags even minor variances for thorough review",
        }
        _VOL_DESCRIPTIONS = {
            "Very low": "only extreme, unprecedented swings are flagged",
            "Low": "only large, clearly unusual swings are flagged",
            "Medium": "flags accounts that moved well outside their "
                      "usual range",
            "High": "catches moderate swings compared to recent history",
            "Very high": "flags any noticeable departure from the usual "
                         "pattern",
        }
        info_mat, info_vol = st.columns(2)
        with info_mat:
            st.caption("**{}:** {}.".format(
                mat_level, _MAT_DESCRIPTIONS[mat_level]))
        with info_vol:
            st.caption("**{}:** {}.".format(
                vol_level, _VOL_DESCRIPTIONS[vol_level]))

        with st.expander("View effective thresholds"):
            _thresh_rows = [{
                "Account": "Global default",
                "% band": "{:.1%}".format(eff_gt["pct_band"]),
                "Min floor": "{}{:,.0f}".format(
                    CURRENCY_SYMBOL, eff_gt["min_dollar_floor"]),
                "Big move": "{}{:,.0f}".format(
                    CURRENCY_SYMBOL, eff_gt["big_dollar"]),
            }]
            for _acct in sorted(eff_at.keys()):
                _t = eff_at[_acct]
                _thresh_rows.append({
                    "Account": _acct,
                    "% band": "{:.1%}".format(_t["pct_band"]),
                    "Min floor": "{}{:,.0f}".format(
                        CURRENCY_SYMBOL, _t["min_dollar_floor"]),
                    "Big move": "{}{:,.0f}".format(
                        CURRENCY_SYMBOL, _t["big_dollar"]),
                })
            st.dataframe(
                pd.DataFrame(_thresh_rows),
                use_container_width=True, hide_index=True)
            st.caption(
                "Near-zero floor: {}{:,.0f} (accounts below this base "
                "are compared by amount only, not percentage).".format(
                    CURRENCY_SYMBOL, eff_nz))
            st.caption(
                "Want to set your own numbers? Switch to "
                "**Custom thresholds** above.")

    else:
        st.caption(
            "Edit any cell to customise an account's thresholds. "
            "The flagged table updates instantly.")

        _all_accounts = list(close_df["account"].unique())
        _floor_col = "Min floor ({})".format(CURRENCY_SYMBOL)
        _big_col = "Big move ({})".format(CURRENCY_SYMBOL)
        _thresh_data = []
        for _acct in _all_accounts:
            _base = ACCOUNT_THRESHOLDS.get(_acct, GLOBAL_THRESHOLD)
            _thresh_data.append({
                "Account": _acct,
                "% band": _base["pct_band"] * 100,
                _floor_col: float(_base["min_dollar_floor"]),
                _big_col: float(_base["big_dollar"]),
            })

        _edited = st.data_editor(
            pd.DataFrame(_thresh_data),
            use_container_width=True,
            hide_index=True,
            disabled=["Account"],
            column_config={
                "% band": st.column_config.NumberColumn(
                    format="%.1f", min_value=0.1, step=1.0,
                    help="Percentage variance that triggers a flag"),
                _floor_col: st.column_config.NumberColumn(
                    format="%.0f", min_value=0.0, step=1000.0,
                    help="Minimum currency amount for the percentage "
                         "test to apply"),
                _big_col: st.column_config.NumberColumn(
                    format="%.0f", min_value=0.0, step=5000.0,
                    help="Single-move amount that flags regardless "
                         "of percentage"),
            },
            key="pa_table",
        )

        eff_gt = GLOBAL_THRESHOLD.copy()
        eff_at = {}
        for _, _row in _edited.iterrows():
            eff_at[_row["Account"]] = {
                "pct_band": _row["% band"] / 100,
                "min_dollar_floor": _row[_floor_col],
                "big_dollar": _row[_big_col],
            }

        zc1, zc2 = st.columns(2)
        eff_zc = zc1.number_input(
            "Volatility cutoff",
            value=MODIFIED_Z_CUTOFF, format="%.1f",
            step=0.5, min_value=0.5, key="pa_zc",
            help="How far a value must deviate from the account's usual "
                 "range to be flagged. Lower values catch smaller swings.")
        eff_nz = zc2.number_input(
            "Near-zero floor ({})".format(CURRENCY_SYMBOL),
            value=float(NEAR_ZERO), format="%.0f",
            step=100.0, min_value=0.0, key="pa_nz",
            help="Accounts with a base below this amount are compared "
                 "by currency amount only, not percentage.")


    # =========================================================================
    # LIVE DETECTION
    # =========================================================================

    flagged, flags = detect_anomalies(
        close_df, history,
        account_thresholds=eff_at,
        global_threshold=eff_gt,
        z_cutoff=eff_zc,
        near_zero=eff_nz,
    )

    flagged_names = {f["account"] for f in flagged}
    flagged_lookup = {f["account"]: f for f in flagged}


    # =========================================================================
    # FLAGGED TABLE: all eight accounts, flagged ones highlighted
    # =========================================================================

    st.markdown("#### Scan results")

    n_flagged = len(flagged)
    st.markdown("**{} of {} accounts flagged.**".format(n_flagged, len(close_df)))

    # Build the full result for every account (flagged or not).
    result_rows = []
    for _, row in close_df.iterrows():
        account = row["account"]
        actual  = row["actual"]
        is_flag = account in flagged_names
        f       = flagged_lookup.get(account)

        bench_parts = []
        for bench in BENCHMARKS:
            if f:
                c = f["comparisons"][bench]
                dollar = c["dollar"]
                pct    = c["pct"]
            else:
                dollar = actual - row[bench]
                pct    = (dollar / row[bench]) if abs(row[bench]) >= eff_nz else None

            pct_str = "{:+.1%}".format(pct) if pct is not None else "n/a"
            bench_parts.append("{} ({})".format(format_money(dollar, signed=True), pct_str))

        if f:
            vol_label = f["volatility"]
            reason_str = "; ".join(flag_reasons(f))
        else:
            vol_label  = ""
            reason_str = ""

        result_rows.append({
            "Account":    account,
            "Actual":     format_money(actual),
            "vs Prior":   bench_parts[0],
            "vs Budget":  bench_parts[1],
            "vs Forecast": bench_parts[2],
            "Volatility": vol_label.replace("_", " ").capitalize(),
            "Reason":     reason_str,
            "_flagged":   is_flag,
            "_stable_moved": f["stable_moved"] if f else False,
        })

    result_df = pd.DataFrame(result_rows)


    def _highlight_flagged(row):
        """Row-level styling: light red for flagged, default for unflagged."""
        if row["_flagged"]:
            return ["background-color: #FFF0F0"] * len(row)
        return [""] * len(row)


    show_cols = ["Account", "Actual", "vs Prior", "vs Budget", "vs Forecast",
                 "Volatility", "Reason"]
    styled = (result_df[show_cols + ["_flagged"]]
              .style
              .apply(_highlight_flagged, axis=1)
              .set_properties(subset=["Account"], **{"font-weight": "bold"}))

    # Hide the internal _flagged column from display.
    styled = styled.hide(subset=["_flagged"], axis="columns")

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Volatility": st.column_config.TextColumn(width="medium"),
            "Reason":     st.column_config.TextColumn(width="large"),
        },
    )

    # Honest caption for stable-moved accounts.
    has_stable = any(r["_stable_moved"] for r in result_rows)
    if has_stable:
        st.caption(
            "Accounts marked \"stable account moved\" have a flat history "
            "(every month identical) and their current value differs. This is a "
            "property of the data, not controlled by either slider. A fixed line "
            "that moves is always worth a look.")


# =============================================================================
# SHARED CARD RENDERER (used by both live and example paths)
# =============================================================================

def _render_narrative(text):
    """Render a narrative fragment in the portfolio callout style."""
    import html as _html
    import re
    safe = _html.escape(text)
    safe = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', safe)
    for priority in ("HIGH", "MEDIUM", "LOW"):
        safe = safe.replace(
            "[{}]".format(priority), priority_badge_html(priority))
    safe = safe.replace("\n", "<br>")
    st.markdown(
        '<div class="sc-narrative">{}</div>'.format(safe),
        unsafe_allow_html=True,
    )


def _drop_rule_lines(text):
    """Remove horizontal-rule lines ("---", "***", "___"), which a callout
    would show as literal text. The narrative ends with one."""
    import re
    return "\n".join(
        line for line in text.split("\n")
        if not re.fullmatch(r"\s*([-*_])(\s*\1){2,}\s*", line)).strip()


def _split_narrative(narrative):
    """Split the narrative into summary and recommended actions.

    The per-account prose is skipped because the triage cards present
    the same information in a more structured format.
    """
    sections = [_drop_rule_lines(s) for s in narrative.split("\n---\n")]
    sections = [s for s in sections if s]
    summary = sections[0] if sections else ""
    if summary.startswith("# "):
        newline = summary.find("\n")
        if newline >= 0:
            summary = summary[newline:].strip()
    actions = ""
    for section in sections[1:]:
        if "RECOMMENDED ACTIONS" in section:
            actions = section
            break
    return summary, actions


def _render_triage_cards(matched, missing):
    import html as _html
    sorted_entries = sort_by_priority(list(matched.values()))
    for entry in sorted_entries:
        fields = [
            ("Headline", entry.get("headline", "")),
            ("Assessment", entry.get("assessment", "")),
            ("Story", entry.get("story", "")),
            ("Hypothesis", entry.get("hypothesis", "")),
            ("Confidence", entry.get("confidence", "")),
        ]
        body_html = "".join(
            '<p class="sc-card-field">'
            '<span class="sc-card-label">{}</span> '
            '{}</p>'.format(label, _html.escape(str(value)))
            for label, value in fields
        )
        with st.container(border=True):
            st.markdown(
                '<div class="sc-card-header">'
                '{badge} '
                '<span class="sc-card-account">{account}</span>'
                '</div>'
                '{body}'.format(
                    badge=priority_badge_html(
                        entry.get("priority", "LOW")),
                    account=escape_text(entry.get("account", "")),
                    body=body_html,
                ),
                unsafe_allow_html=True,
            )

    for name in missing:
        with st.container(border=True):
            st.markdown(
                '<div class="sc-card-header">'
                '<span class="sc-card-account">{account}</span>'
                '</div>'.format(account=escape_text(name)),
                unsafe_allow_html=True,
            )
            st.caption(
                "No assessment returned by the model for this account.")


# =============================================================================
# SIGN-OFF (after Project 6: the rule and its messages live in src/signoff.py)
# =============================================================================

# The sign-off that stands (a SignOff) and the result it was signed for (a
# fingerprint), the two boxes, the name kept while the section is not drawn,
# the result the SIGN box was typed for, and the last refusal
_SIGNED_OFF_KEY = "cs_signed_off"
_SIGNED_FOR_KEY = "cs_signed_for"
_REVIEWER_KEY = "cs_reviewer"
_NAME_KEPT_KEY = "cs_reviewer_kept"
_SIGN_KEY = "cs_sign"
_SIGN_FOR_KEY = "cs_sign_for"
_REFUSAL_KEY = "cs_signoff_refusal"

_SIGN_MEANS = ("Signing says you have read the result against the figures "
               "above. The figures come from Python. This version does not "
               "check the numbers in the model's wording.")


def _thresholds_key(eff_at, eff_gt, eff_zc, eff_nz):
    """The effective thresholds as one comparable text: those a triage was
    run at, or those on screen now (the ones the PDF prints)."""
    return json.dumps({"account_thresholds": eff_at, "global_threshold": eff_gt,
                       "z_cutoff": eff_zc, "near_zero": eff_nz},
                      sort_keys=True, default=float)


def _result_fingerprint(state, flagged, thresholds):
    """The result as the live PDF prints it, as one hash: the run (its number
    and the model's triage), the flagged accounts with their figures, and
    the effective thresholds. A sign-off stands only while this is
    unchanged."""
    text = json.dumps({
        "run": state["run"], "narrative": state["narrative"],
        "matched": state["matched"], "missing": state["missing"],
        "tokens": [state["tok_in"], state["tok_out"]],
        "stop_reason": state["stop_reason"],
        "flagged": flagged, "thresholds": thresholds,
    }, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _standing_signoff(result):
    """The sign-off that stands for this result, or None. One made for
    another result no longer applies: this is a new DRAFT, the name stays
    filled in and the SIGN box is cleared. The SIGN box is also cleared
    whenever the result it was typed for changes (a new run)."""
    ss = st.session_state
    signed = ss.get(_SIGNED_OFF_KEY)
    if signed is not None and ss.get(_SIGNED_FOR_KEY) != result:
        del ss[_SIGNED_OFF_KEY]
        del ss[_SIGNED_FOR_KEY]
        ss[_NAME_KEPT_KEY] = signed.name
        signed = None
    if ss.get(_SIGN_FOR_KEY) != result:
        ss[_SIGN_KEY] = ""
        ss[_SIGN_FOR_KEY] = result
    return signed


def _keep_name():
    """Keep the typed name under a key of its own: a box that is not drawn
    (no triage on screen, or a FINAL) loses its value."""
    st.session_state[_NAME_KEPT_KEY] = st.session_state.get(_REVIEWER_KEY, "")


def _sign_off(run, result, outstanding):
    """The Sign off button (on_click, as in Project 6). The rule decides; the
    page keeps the sign-off, the result it was signed for, and adds it to
    the run's row in the session audit."""
    ss = st.session_state
    signed, refusal = sign_off(ss.get(_REVIEWER_KEY), ss.get(_SIGN_KEY),
                               already_signed=_SIGNED_OFF_KEY in ss)
    if refusal:
        ss[_REFUSAL_KEY] = refusal
        return
    ss[_SIGNED_OFF_KEY] = signed
    ss[_SIGNED_FOR_KEY] = result
    sign_run(run, signed, outstanding)


def _download_pdf(signed, pdf, pdf_error, file_name):
    """The review PDF: the DRAFT PDF, or the FINAL PDF once signed. If it
    could not be built, its plain message shows in its place."""
    if pdf is None:
        st.error("The review PDF could not be built. {}".format(pdf_error))
        return
    st.download_button(
        label=("Download the FINAL PDF" if signed is not None
               else "Download the DRAFT PDF"),
        data=pdf,
        file_name=file_name,
        mime="application/pdf",
    )


def _render_signoff(run, result, signed, stale, outstanding, *, pdf,
                    pdf_error, pdf_name):
    """The Sign-off section: DRAFT with the boxes and the button, or FINAL
    and read only, each with its PDF in Project 4's order. Stale (the
    thresholds on screen are not those the triage was run at) keeps the
    button disabled."""
    ss = st.session_state
    refusal = ss.pop(_REFUSAL_KEY, None)

    st.markdown("---")
    st.markdown("#### Sign-off")
    st.markdown(_SIGN_MEANS)

    if signed is not None:
        st.markdown("{}&nbsp; {}".format(
            status_chip_html(FINAL), html.escape(signed.line)),
            unsafe_allow_html=True)
        if refusal:
            st.warning(refusal)
        _download_pdf(signed, pdf, pdf_error, pdf_name)
        st.caption(SCOPE_NOTE)
        return

    st.markdown(status_chip_html("draft"), unsafe_allow_html=True)
    if outstanding:
        st.markdown("Outstanding before you sign:")
        st.markdown("\n".join("{}. {}".format(i, line) for i, line
                              in enumerate(outstanding, start=1)))
    else:
        st.markdown("Nothing is outstanding.")
    if _NAME_KEPT_KEY in ss:
        ss[_REVIEWER_KEY] = ss[_NAME_KEPT_KEY]
    st.text_input("Reviewer name", key=_REVIEWER_KEY, on_change=_keep_name)
    st.text_input(WORD_LABEL, key=_SIGN_KEY, help=WORD_HELP)
    st.button("Sign off", key="cs_signoff", type="primary", on_click=_sign_off,
              args=(run, result, outstanding), disabled=stale)
    if stale:
        st.caption(STALE_CAPTION)
    st.caption(SCOPE_NOTE)
    if refusal:
        st.warning(refusal)
    _download_pdf(None, pdf, pdf_error, pdf_name)


# =============================================================================
# KEYLESS EXAMPLE (cached)
# =============================================================================

@st.cache_data
def _load_example():
    return load_example()

example = _load_example()


with tab_live:
    st.markdown("---")

    # =========================================================================
    # AI TRIAGE: the live path (the worked example has its own tab)
    # =========================================================================

    st.markdown("#### AI triage")
    st.caption(
        "Run the AI agent to triage the flagged accounts: rank by priority, "
        "judge timing versus permanent, and hypothesise a likely cause. "
        "One model call per click.")

    fingerprint = triage_fingerprint(flagged)

    # Stale check: clear saved triage if thresholds changed since it ran.
    _stale_cleared = False
    if "triage_state" in st.session_state:
        if st.session_state["triage_state"]["fingerprint"] != fingerprint:
            del st.session_state["triage_state"]
            _stale_cleared = True

    if _stale_cleared:
        st.info(
            "The flagged set changed. Run AI triage again to interpret "
            "the current accounts.")

    if n_flagged == 0:
        st.info(
            "No accounts flagged at the current thresholds. "
            "Nothing to triage.")
    elif client is None:
        st.info(
            "Paste your Anthropic API key in the sidebar to run "
            "a live triage.")
    else:
        run_clicked = st.button("Run AI triage", key="run_triage",
                                type="primary")
        if run_clicked:
            with st.spinner("Running AI triage..."):
                try:
                    system_prompt, user_prompt = build_prompt(
                        flagged, DEFAULT_ENTITY, CLOSE_PERIOD)
                    text, tok_in, tok_out, stop_reason = call_claude(
                        system_prompt, user_prompt, client=client)
                except (RuntimeError, Exception) as exc:
                    st.error(friendly_message(exc))
                else:
                    narrative = strip_json_block(text)
                    triage_json = parse_triage_json(text)
                    if triage_json is None:
                        matched = {}
                        missing_names = [f["account"] for f in flagged]
                    else:
                        matched, missing_names = match_triage_to_flagged(
                            triage_json, flagged)
                    record_run(
                        flagged,
                        (eff_at, eff_gt, eff_zc, eff_nz),
                        file_sha256(CLOSE_FILE),
                        file_sha256(HISTORY_FILE),
                        tok_in, tok_out, stop_reason,
                    )
                    # The run's number in the session audit list, and the
                    # thresholds it was run at, for the Sign-off section
                    st.session_state["triage_state"] = {
                        "fingerprint": fingerprint,
                        "narrative": narrative,
                        "matched": matched,
                        "missing": missing_names,
                        "tok_in": tok_in,
                        "tok_out": tok_out,
                        "stop_reason": stop_reason,
                        "run": len(get_runs()),
                        "thresholds": _thresholds_key(
                            eff_at, eff_gt, eff_zc, eff_nz),
                    }

        if "triage_state" in st.session_state:
            state = st.session_state["triage_state"]

            if state["stop_reason"] == "max_tokens":
                st.warning(
                    "The model's response was truncated. The triage may be "
                    "incomplete. Consider raising MAX_TOKENS in config.py.")

            _summary = ""
            _actions = ""
            if state["narrative"]:
                _summary, _actions = _split_narrative(state["narrative"])
                if _summary:
                    _render_narrative(_summary)

            if state["missing"]:
                st.warning(
                    "The model did not return an assessment for: {}. "
                    "These accounts are flagged but have no triage "
                    "below.".format(", ".join(state["missing"])))

            _render_triage_cards(state["matched"], state["missing"])
            if _actions:
                _render_narrative(_actions)
            st.caption(format_cost(state["tok_in"], state["tok_out"]))

            # The sign-off that stands for this result, decided before the
            # PDF is built. The triage on screen always matches the flags on
            # screen (a changed flagged set clears it above); it is stale
            # when the thresholds are no longer those it was run at.
            _thresholds = _thresholds_key(eff_at, eff_gt, eff_zc, eff_nz)
            _result = _result_fingerprint(state, flagged, _thresholds)
            _signed = _standing_signoff(_result)
            _stale = state["thresholds"] != _thresholds

            _dl_triage = list(state["matched"].values())
            # In UTC, as the signed line is, whatever the server's own zone
            _dl_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
            # A PDF that fails to build shows a plain message in the Sign-off
            # section; the CSV still downloads.
            _dl_pdf_error = None
            try:
                _dl_pdf = build_pdf_bytes(
                    state["narrative"], flagged, _dl_triage,
                    state["tok_in"], state["tok_out"],
                    effective_thresholds=(eff_at, eff_gt, eff_zc, eff_nz),
                    status_line=pdf_status_line(_signed))
            except Exception as exc:
                _dl_pdf = None
                _dl_pdf_error = friendly_message(exc)
            _dl_csv = build_csv_bytes(flagged, _dl_triage)
            # The data stays a plain download; the review PDF is in Sign-off.
            st.download_button(
                label="Download anomalies CSV",
                data=_dl_csv,
                file_name="anomalies_{}.csv".format(_dl_ts),
                mime="text/csv",
            )

            _render_signoff(state["run"], _result, _signed, _stale,
                            outstanding_lines(flagged), pdf=_dl_pdf,
                            pdf_error=_dl_pdf_error,
                            pdf_name="close_review_{}.pdf".format(_dl_ts))

with tab_example:
    if fingerprint == example["fingerprint"]:
        st.caption(
            "This is a captured model output at the default sensitivity, as "
            "the model wrote it except for edits made afterwards: \"EUR\" was "
            "added before three amounts, and \"absolute terms\" became \"euro "
            "terms\". Enter an API key in the sidebar to run live triage.")
        _ex_summary = ""
        _ex_actions = ""
        if example["narrative"]:
            _ex_summary, _ex_actions = _split_narrative(
                example["narrative"])
            if _ex_summary:
                _render_narrative(_ex_summary)
        _render_triage_cards(example["matched"], example["missing"])
        if _ex_actions:
            _render_narrative(_ex_actions)
        st.caption(format_cost(example["tokens_in"], example["tokens_out"]))
    elif n_flagged == 0:
        st.info(
            "No accounts flagged at the current thresholds. "
            "Nothing to triage.")
    else:
        # The controls are in the other tab, so the hint names it.
        if threshold_mode == "Sensitivity sliders":
            restore_hint = "Move the sliders in Run it yourself back to Medium"
        else:
            restore_hint = ("Restore the default custom threshold values in "
                            "Run it yourself")
        st.info(
            "The captured triage was produced at the default sensitivity. "
            "{} to see it, or enter an API "
            "key in the sidebar to run live triage.".format(restore_hint))

# -- Session audit -----------------------------------------------------------
audit_runs = get_runs()
if audit_runs:
    n_runs = len(audit_runs)
    with st.expander(
            "Session audit ({} run{})".format(
                n_runs, "s" if n_runs != 1 else "")):
        st.caption(
            "Session-only governance record. "
            "Cleared when the browser tab closes.")
        for i, record in enumerate(audit_runs):
            ts_display = record["timestamp"][:19].replace("T", " ") + " UTC"
            st.markdown("**Run {}** -- {}".format(n_runs - i, ts_display))
            st.markdown("{} account{} flagged: {}".format(
                record["n_flagged"],
                "s" if record["n_flagged"] != 1 else "",
                ", ".join(record["flagged_accounts"])))
            gt = record["effective_thresholds"]["global_threshold"]
            st.caption(
                "Global: {:.1%} band, {}{:,.0f} floor, {}{:,.0f} big move; "
                "z-cutoff {:.1f}; near-zero {}{:,.0f}".format(
                    gt["pct_band"],
                    CURRENCY_SYMBOL, gt["min_dollar_floor"],
                    CURRENCY_SYMBOL, gt["big_dollar"],
                    record["effective_thresholds"]["z_cutoff"],
                    CURRENCY_SYMBOL,
                    record["effective_thresholds"]["near_zero"]))
            st.caption("`{}`  `{}`".format(
                record["close_hash"][:30] + "...",
                record["history_hash"][:30] + "..."))
            st.caption(format_cost(record["tokens_in"], record["tokens_out"]))
            if record["requires_review"]:
                st.caption("Requires human review: **yes**")
            if record.get("status") == FINAL:
                _row_signed = SignOff(
                    record["signed_off_by"],
                    datetime.fromisoformat(record["signed_off_at"]))
                st.markdown("{}&nbsp; {}".format(
                    status_chip_html(FINAL), html.escape(_row_signed.line)),
                    unsafe_allow_html=True)
            if i < n_runs - 1:
                st.divider()

st.divider()
st.caption("Sample data only. All figures are illustrative.")
