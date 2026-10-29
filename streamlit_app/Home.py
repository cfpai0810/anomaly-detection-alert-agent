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

from streamlit_app.lib.theme import inject_css, header_html
from streamlit_app.lib.key_gate import render_key_gate


def _home_page():
    """Render the landing page content."""
    st.markdown(inject_css(), unsafe_allow_html=True)

    # -- Header ---------------------------------------------------------------
    st.markdown(header_html(
        "Anomaly Detection and Alert Agent",
        "A month-end close anomaly scanner and triage aid for a finance "
        "controller."),
        unsafe_allow_html=True)

    # -- One step to the worked example, which needs no key -------------------
    # The button only switches page: Close scan opens on its first tab, the
    # worked example.
    if st.button("See the worked example", key="see_worked_example"):
        st.switch_page(CLOSE_SCAN_PAGE)

    # -- The key gate (sidebar): returns a per-session client or None ---------
    client = render_key_gate()
    st.session_state["live_client"] = client

    # -- What it does ---------------------------------------------------------
    st.markdown("#### What it does")
    st.markdown(
        "- Reads a month-end close: eight accounts, each with actual, prior "
        "month, budget, forecast and twelve months of history.\n"
        "- Flags unusual accounts by rule, with sensitivity sliders you "
        "control.\n"
        "- On request, the model triages each flag: timing or permanent, "
        "which benchmark matters, what might have caused it.\n"
        "- After a live triage, produces a review PDF and an anomalies CSV "
        "for the controller. The PDF is a DRAFT until a named reviewer signs "
        "it.")

    # -- The governance promise -----------------------------------------------
    st.markdown("#### The governance promise")
    st.markdown(
        "- **Python detects.** Flags come from a materiality test against "
        "three benchmarks and a volatility test, calculated in code.\n"
        "- **A rule flags.** The thresholds are yours, not the model's.\n"
        "- **The model triages.** It interprets flags it was given and "
        "cannot change them.\n"
        "- **The reviewer signs.** Nothing is final until a named reviewer reads it "
        "and signs.")

    # -- Pages ----------------------------------------------------------------
    st.markdown("#### Pages")
    st.page_link(HOW_IT_WORKS_PAGE, label=(
        "How it works: the two-stage model, what flags an account, what the "
        "sliders do, and what the output means for a reviewer"))
    st.page_link(CLOSE_SCAN_PAGE, label=(
        "Close scan: a worked example that needs no key, and a scan of the "
        "sample close at the sensitivity you choose"))
    st.page_link(OWN_DATA_PAGE, label=(
        "Your own data: the two data files, how to run the scanner locally, "
        "and what stays private"))
    st.page_link(HOW_ITS_BUILT_PAGE, label=(
        "How it's built: the Python-detects, model-triages boundary, the "
        "pipeline, determinism and lineage, and the review stance"))

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
            "million output tokens, one triage costs about 1.8 US cents: the "
            "worked example's run used 1,191 input and 930 output tokens.")
        st.caption(
            "Your key is used only for your session and is not stored by "
            "this site. On the web the data stays sample-only; to work on "
            "your own numbers, run the project locally from GitHub.")

    st.divider()
    st.caption("Sample data only: Valencia Operations is a fictional company. "
               "All figures are illustrative.")

    # -- Sample-data status: a warning only if a file is missing ---------------
    DATA_DIR = ROOT / "data"
    expected = ["close_actuals.csv", "account_history.csv"]
    missing = [f for f in expected if not (DATA_DIR / f).exists()]
    if missing:
        st.warning(
            "The sample data is incomplete. Missing: " + ", ".join(missing) +
            ". The tool page needs these files in the project's data folder.")


# -- Multipage navigation (Streamlit >= 1.36) ---------------------------------
st.set_page_config(
    page_title="Anomaly Detection Agent",
    page_icon=":material/search:",
    layout="centered",
    initial_sidebar_state="auto",
)

# The view files live in a folder NOT named "pages", so Streamlit's automatic
# pages/ navigation never triggers and st.navigation() below is the sole nav
# (with these explicit titles and this order, not filename-derived alphabetical),
# even when a page's address is the first request to a freshly started app.
# Each page's address is pinned with url_path: the addresses are published, so
# neither a new title nor a renamed file may move them. A test lists them all.
VIEWS_DIR = Path(__file__).resolve().parent / "views"

HOW_IT_WORKS_PAGE = st.Page(str(VIEWS_DIR / "how_it_works.py"),
                            url_path="how_it_works",
                            title="How it works", icon=":material/menu_book:")
# The worked example lives on this page, in its first tab.
CLOSE_SCAN_PAGE = st.Page(str(VIEWS_DIR / "close_scan.py"),
                          url_path="close_scan",
                          title="Close scan", icon=":material/search:")
OWN_DATA_PAGE = st.Page(str(VIEWS_DIR / "your_own_data.py"),
                        url_path="your_own_data",
                        title="Your own data", icon=":material/upload_file:")
HOW_ITS_BUILT_PAGE = st.Page(str(VIEWS_DIR / "how_its_built.py"),
                             url_path="how_its_built",
                             title="How it's built", icon=":material/build:")

pg = st.navigation([
    st.Page(_home_page, title="Home", icon=":material/home:"),
    HOW_IT_WORKS_PAGE,
    CLOSE_SCAN_PAGE,
    OWN_DATA_PAGE,
    HOW_ITS_BUILT_PAGE,
])

pg.run()
