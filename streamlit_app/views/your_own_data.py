# =============================================================================
# views/your_own_data.py: running the scanner on your own data
# =============================================================================
# Renders docs/bring_your_own_data.md in Streamlit. The doc is the
# single source of truth; this page just reads and displays it.
# =============================================================================

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit_app.lib.theme import inject_css, header_html

st.markdown(inject_css(), unsafe_allow_html=True)

DOC_PATH = ROOT / "docs" / "bring_your_own_data.md"
DOC_TITLE = "### Your own data"

body = DOC_PATH.read_text(encoding="utf-8")

# The banner carries the page's name, so the document's own title line is not
# drawn a second time under it.
st.markdown(header_html(
    "Your own data",
    "The two files the scanner reads, and how to run it on your own "
    "month-end close."),
    unsafe_allow_html=True)
st.markdown(body.removeprefix(DOC_TITLE).lstrip())

st.divider()
st.caption("Sample data only. All figures are illustrative.")
