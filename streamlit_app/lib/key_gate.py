# =============================================================================
# lib/key_gate.py: the bring-your-own-key gate (foundation component)
# =============================================================================
# Renders the API-key input, holds the key ONLY in session state for the live
# session, and builds a per-session Anthropic client. The key is held in the
# server's memory for that session only: it is never written to disk and
# never logged. This is reused by every tool page.
#
# Honest disclosure (per the master plan): "held in the server's memory for
# your session only, never stored", NOT "never leaves your device", because
# the key does reach the server to make the Claude call. Users who want their
# key to never leave their machine are pointed to the GitHub download.
# =============================================================================

import streamlit as st

from streamlit_app.lib.claude_client import build_client, looks_like_anthropic_key
from streamlit_app.lib.errors import friendly_message, KEY_LOOKS_COMPLETE

_KEPT = "anthropic_api_key"   # survives page changes; the widget's own state does not


def _keep_key():
    st.session_state[_KEPT] = st.session_state.get("byo_api_key", "")


# Constant on purpose: a changed label resets the expander.
GATE_LABEL = "Run it live with your own key (optional)"
LIVE_ON = "Live mode is on for this session."
LIVE_OFF = "Live mode is off. The worked examples need no key."


def render_key_gate():
    """Render the key gate in the sidebar, in a closed expander, with one line
    below it saying whether live mode is on. Returns the per-session client,
    or None if no valid key is present yet. Safe to call on every page."""
    with st.sidebar.expander(GATE_LABEL, expanded=False):
        client = _gate_body()
    st.sidebar.caption(LIVE_ON if client is not None else LIVE_OFF)
    return client


def _gate_body():
    """The gate's contents: the caption, the key input, the billing sentence,
    the privacy note and any warning. Returns the client or None."""
    st.caption(
        "Paste your Anthropic API key to run a live analysis. "
        "Don't have one? Use the example output on the main page, no key needed.")

    # The key lives ONLY in session state. Password field masks it on entry.
    # Streamlit drops a widget's state when another page runs, so the key is
    # kept under a non-widget key and the widget is seeded from it each run.
    if _KEPT in st.session_state:
        st.session_state["byo_api_key"] = st.session_state[_KEPT]
    key = st.text_input(
        "Anthropic API key",
        type="password",
        placeholder="sk-ant-...",
        key="byo_api_key",
        on_change=_keep_key,
        help="Held in the server's memory for this session only. Never stored, never logged.",
    )

    st.caption(
        "Running a live analysis uses your own Anthropic API key and bills "
        "your Anthropic account a small amount per run. The worked examples "
        "are free and need no key.")

    # Honest disclosure, always visible inside the gate.
    st.markdown(
        '<div class="sc-keynote">Your key is held in the server\'s memory for '
        'your session only, used to make the call, and never stored or logged. '
        'If you would rather your key never leave your own machine, download '
        'the project from GitHub and run it locally.</div>',
        unsafe_allow_html=True,
    )

    if not key:
        return None

    # Cheap client-side sanity check before we build anything.
    if not looks_like_anthropic_key(key):
        st.warning(
            "That does not look like a complete Anthropic key. It should start "
            "with sk-ant- and be the full string.")
        return None

    # Build a per-session client. Never global, never shared across users.
    try:
        client = build_client(key)
    except Exception as exc:
        st.error(friendly_message(exc))
        return None

    st.success(KEY_LOOKS_COMPLETE)
    return client


def has_live_client(client):
    """Small helper for pages to branch on live vs example mode."""
    return client is not None
