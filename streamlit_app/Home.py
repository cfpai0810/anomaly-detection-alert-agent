# =============================================================================
# Home.py: the app shell and landing page (Streamlit entry point)
# =============================================================================
# Run locally: streamlit run streamlit_app/Home.py
# Streamlit >= 1.36 uses st.navigation + st.Page for multipage apps.
# =============================================================================

import sys
sys.dont_write_bytecode = True

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit_app.lib.theme import inject_css, DARK_BLUE
from streamlit_app.lib.key_gate import render_key_gate


def _home_page():
    """Render the landing page content."""
    st.markdown(inject_css(), unsafe_allow_html=True)

    # -- Header ---------------------------------------------------------------
    st.markdown(
        '<div class="sc-header">'
        '<h1>Anomaly Detection and Alert Agent</h1>'
        '<p>A month-end close anomaly scanner and triage aid for a finance '
        'controller. It scans one entity\'s close for unusual accounts and '
        'helps the controller focus review time on what matters.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # -- The key gate (sidebar): returns a per-session client or None ---------
    client = render_key_gate()
    st.session_state["live_client"] = client

    # -- What this does -------------------------------------------------------
    st.markdown("#### What this does")
    st.write(
        "The tool works in two stages. First, it flags unusual accounts "
        "with transparent, deterministic rules you control: a materiality "
        "test against three benchmarks (prior month, budget, forecast) and "
        "a statistical volatility test against the account's own history. "
        "Then, on request, an AI layer interprets each flag: is the "
        "movement timing or permanent, which benchmark tells the real "
        "story, and what might have caused it.")
    st.write(
        "The scan runs without an API key. The sensitivity sliders, the "
        "flagged table, and a worked example with real model output are "
        "all free to explore. Only the live AI triage needs an Anthropic "
        "API key.")

    st.markdown("---")

    # -- Pages ----------------------------------------------------------------
    st.markdown("#### Pages")
    st.write(
        "**How it works** explains the tool in plain terms: the two-stage "
        "model, what flags an account, what the sliders do, and what the "
        "output means for a reviewer.")
    st.write(
        "**Close anomaly scan** is the tool itself. It loads the sample "
        "close, lets you adjust the detection sensitivity, flags the "
        "anomalies, and triages them. A worked example runs at the "
        "default sensitivity with no API key.")
    st.write(
        "**How it's built** is the technical and governance story: the "
        "Python-detects, model-triages boundary, the pipeline, "
        "determinism and lineage, and the review stance.")
    st.write(
        "**Your own data** explains how to run the scanner on your own "
        "month-end close locally: the two data files, how to run it, "
        "and what stays private.")

    st.markdown("---")

    # -- Sample-data status ---------------------------------------------------
    st.markdown("#### The sample company")
    st.write(
        "Everything here runs on Valencia Operations, a synthetic company "
        "invented for the demo. Its close data and account history describe "
        "a month-end close for June 2026, with seven accounts and twelve "
        "months of trailing history. Because the company and its numbers are "
        "entirely fictional, nothing you do on this site touches real or "
        "personal information.")

    DATA_DIR = ROOT / "data"
    expected = ["close_actuals.csv", "account_history.csv"]
    present = [f for f in expected if (DATA_DIR / f).exists()]
    if len(present) == len(expected):
        st.success(
            "Sample dataset loaded, {} of {} files present. "
            "All figures on this site are illustrative.".format(
                len(present), len(expected)))
    else:
        missing = [f for f in expected if f not in present]
        st.warning(
            "The sample data is incomplete. Missing: " + ", ".join(missing) +
            ". The tool page needs these files in the project's data folder.")

    # -- Using your own API key -----------------------------------------------
    with st.expander("Using your own API key"):
        st.write(
            "The worked example on the scan page is free and needs no key. "
            "To run a live triage with the AI agent, you supply your own "
            "Anthropic API key, and each run bills your Anthropic account "
            "directly. This is not a free AI service; it is your key and "
            "your usage.")
        st.write(
            "To get a key, sign up at the Anthropic Console "
            "(console.anthropic.com), add a payment method under Settings, "
            "then open Settings and API keys and create a key. The key is "
            "shown once, so copy it when it is created. Paste it into the "
            "sidebar here to run a live analysis.")
        st.write(
            "What it costs. Each run makes one short call to Claude Sonnet "
            "to triage the flagged anomalies. At Sonnet's current rate of "
            "about three US dollars per million input tokens and fifteen per "
            "million output tokens, a single scan is well under one US cent "
            "in practice.")
        st.caption(
            "Your key is used only for your session and is not stored by "
            "this site. On the web the data stays sample-only; to work on "
            "your own numbers, run the project locally from GitHub.")

    st.divider()
    st.caption("Sample data only. All figures are illustrative.")


# -- Multipage navigation (Streamlit >= 1.36) ---------------------------------
st.set_page_config(
    page_title="Anomaly Detection Agent",
    page_icon="•",
    layout="centered",
    initial_sidebar_state="expanded",
)

PAGES_DIR = Path(__file__).resolve().parent / "pages"

pg = st.navigation([
    st.Page(_home_page, title="Home", icon=":material/home:"),
    st.Page(str(PAGES_DIR / "how_it_works.py"),
            title="How It Works", icon=":material/menu_book:"),
    st.Page(str(PAGES_DIR / "close_scan.py"),
            title="Close Scan", icon=":material/search:"),
    st.Page(str(PAGES_DIR / "your_own_data.py"),
            title="Your Own Data", icon=":material/upload_file:"),
    st.Page(str(PAGES_DIR / "how_its_built.py"),
            title="How It's Built", icon=":material/build:"),
])

pg.run()
