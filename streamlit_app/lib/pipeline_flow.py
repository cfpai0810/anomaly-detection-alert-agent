# =============================================================================
# lib/pipeline_flow.py: the pipeline diagram on How it's built
# =============================================================================
# The five stages of a scan, in the colours the page has always drawn them in:
# blue for loading, detection and output, amber for the human gate where the
# controller reviews the flags and decides whether to proceed, green for the
# triage that calls the model. Drawn in HTML by lib/flow_diagram.py, so it
# reads on a phone. The renderer has no line above a box, so the three labels
# that sat above Detect, Review flags and Triage are each box's foot line.
# =============================================================================

from streamlit_app.lib.flow_diagram import Role, Stage, flow_html
from streamlit_app.lib.theme import AMBER, AMBER_BG, GREEN, LIGHT_BLUE, MID_BLUE

PIPELINE = Role(MID_BLUE, LIGHT_BLUE)
GATE     = Role(AMBER, AMBER_BG)        # the human gate between detection and triage
TRIAGE   = Role(GREEN, "#EAF3DE")

LABEL = ("The pipeline: load the data files; detect, in Python, deterministic, "
         "free and instant; review flags, a human decision, where you decide to "
         "proceed; triage, by AI on request, which costs tokens; and output the "
         "PDF, CSV and audit. The human gate between detection and triage is "
         "deliberate: the controller reviews the flags before choosing to invoke "
         "the AI.")

STAGES = (
    Stage("Load", "data files", "", PIPELINE),
    Stage("Detect", "Python, deterministic", "free, instant", PIPELINE),
    Stage("Review flags", "human decision", "you decide to proceed", GATE),
    Stage("Triage", "AI, on request", "costs tokens", TRIAGE),
    Stage("Output", "PDF, CSV, audit", "", PIPELINE),
)


def pipeline_flow_html():
    """Return the pipeline diagram as HTML for st.markdown."""
    return flow_html(LABEL, STAGES)
