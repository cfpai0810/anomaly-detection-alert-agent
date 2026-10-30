# =============================================================================
# lib/governance_flow.py -- the signature diagram (Project 3: flag for review)
# =============================================================================
# The flag-for-review governance model as a flow. A person opens and closes
# every run; the deterministic engine detects and flags, the model triages the
# flagged set, and the person reviews. Colour encodes WHO OWNS each step: the
# person (blue), the model for language (purple), the deterministic engine for
# numbers (green). Two consecutive engine boxes reflect the design: detection
# and flagging are separate deterministic steps. Drawn in HTML by
# lib/flow_diagram.py, so it reads on a phone.
# =============================================================================

from streamlit_app.lib.flow_diagram import Role, Stage, flow_html
from streamlit_app.lib.theme import DARK_BLUE, GREEN, LIGHT_BLUE

PERSON = Role(DARK_BLUE, LIGHT_BLUE)
MODEL  = Role("#6B4FA8", "#F0ECF8")     # purple: the model handles language, not numbers
ENGINE = Role(GREEN, "#E8F3E9")

LABEL = ("How a run is governed: you load the month-end close, Python runs the "
         "materiality and z-score tests, accounts that breach the thresholds are "
         "flagged by rule, the model triages the flagged set, and you review every "
         "flagged account, then sign off.")

STAGES = (
    Stage("YOU LOAD THE CLOSE", "A month-end close and its history", "the person",
          PERSON),
    Stage("PYTHON DETECTS", "Materiality and z-score tests", "no model here", ENGINE),
    Stage("FLAGGED BY RULE", "Thresholds you set mark the outliers", "no model here",
          ENGINE),
    Stage("THE MODEL TRIAGES", "Explains each flag, suggests a cause",
          "words, not numbers", MODEL),
    Stage("YOU SIGN OFF", "You judge every flagged account, then sign off",
          "the person", PERSON),
)


def governance_flow_html():
    """Return the governance flow diagram as HTML for st.markdown."""
    return flow_html(
        LABEL, STAGES,
        heading="You open and close every run. The tests and the model work only "
                "in between.",
        note="Every run is written to the audit record, with the thresholds used "
             "and a hash of the data, so it can be traced and reproduced.",
        key=((PERSON, "You"), (MODEL, "The model, for language"),
             (ENGINE, "Deterministic code, for numbers")))
