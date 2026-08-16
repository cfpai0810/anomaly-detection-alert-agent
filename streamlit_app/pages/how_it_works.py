# =============================================================================
# pages/how_it_works.py -- user-facing explanation of the tool
# =============================================================================

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit_app.lib.theme import inject_css

st.markdown(inject_css(), unsafe_allow_html=True)


# -- Title --------------------------------------------------------------------
st.markdown("### How it works")
st.write(
    "A plain-language guide to what the tool does, how it decides what to "
    "flag, and what its output means for you as a reviewer.")

st.markdown("---")


# -- The problem --------------------------------------------------------------
st.markdown("#### The problem")
st.write(
    "At month-end a finance controller reviews every account in the close "
    "for unusual movements. In practice that means scanning a variance "
    "report line by line, comparing actuals against the prior month, the "
    "budget, and the forecast, and deciding which lines need investigation. "
    "With dozens or hundreds of accounts, the risk is not that something "
    "goes unseen but that review time is spread too thinly across lines "
    "that turn out to be routine.")
st.write(
    "This tool automates the scanning step so the controller's time goes "
    "where it matters: the accounts that are genuinely unusual.")

st.markdown("---")


# -- The two stages -----------------------------------------------------------
st.markdown("#### Two stages: detect, then triage")
st.write(
    "The tool works in two stages, and the distinction matters.")

st.markdown("**Stage 1: Detection** (deterministic, instant, free)")
st.write(
    "Python scans every account in the close against three benchmarks "
    "(prior month, budget, and forecast) and against the account's own "
    "trailing history. Accounts that cross a threshold are flagged. This "
    "step uses no AI: the rules are fixed, the maths is transparent, and "
    "the same data always produces the same flags. You can run it as many "
    "times as you like at no cost.")

st.markdown("**Stage 2: Triage** (AI, on request, uses your API key)")
st.write(
    "Once you have the flagged accounts, you can ask the AI agent to "
    "triage them. It reads the variance data Python has already computed "
    "and, for each flagged account, judges whether the movement is likely "
    "timing or permanent, identifies which benchmark tells the most "
    "important story, and hypothesises a likely cause. This step makes one "
    "call to Claude and uses your Anthropic API key.")

st.info(
    "The AI never decides what is suspicious. It only interprets accounts "
    "that Python has already flagged. It cannot remove a flag or hide an "
    "account from the review.")

st.markdown("---")


# -- What flags an account ----------------------------------------------------
st.markdown("#### What flags an account")
st.write(
    "An account is flagged when either of two tests is met.")

st.markdown("**Materiality test.**")
st.write(
    "The variance against any of the three benchmarks is large enough to "
    "matter. \"Large enough\" means the percentage clears a band (for "
    "example 10% for most accounts, tighter for Revenue and Personnel) "
    "and the currency amount clears a floor, or the currency amount alone "
    "is very large regardless of percentage. This stops tiny-base noise "
    "from triggering a flag (a 200% move on a low-value account is not "
    "material) while still catching big moves on large accounts even when "
    "the percentage looks modest.")

st.markdown("**Volatility test.**")
st.write(
    "The current value is statistically unusual compared to the account's "
    "own trailing twelve months. The tool uses a modified z-score (based "
    "on the median rather than the mean), which is robust when one or two "
    "months in the history are themselves outliers. A high z-score means "
    "the current value is far from where this account normally sits.")

st.markdown("**Stable-moved rule.**")
st.write(
    "A special case: if an account has been perfectly flat (every month "
    "identical) and its current value differs at all, it is always flagged. "
    "A fixed line that moves is worth a look regardless of the size of the "
    "move. This rule is a property of the data, not controlled by the "
    "sensitivity sliders.")

st.markdown("---")


# -- The sliders --------------------------------------------------------------
st.markdown("#### Detection sensitivity")
st.write(
    "The detection page offers two ways to set sensitivity, chosen with "
    "a control at the top of the section.")

st.markdown("**Sensitivity sliders.**")
st.write(
    "Two plain sliders: materiality sensitivity and volatility sensitivity. "
    "Each has five levels from \"Very low\" to \"Very high\", with Medium "
    "matching the default thresholds used by the command-line tool. "
    "Higher materiality sensitivity means smaller thresholds, so smaller "
    "variances are caught. Higher volatility sensitivity catches smaller "
    "swings relative to the account's recent history. Moving either slider "
    "re-runs the detection instantly and the flagged table updates live.")

st.markdown("**Custom thresholds.**")
st.write(
    "Set the exact threshold values for each account and the volatility "
    "cutoff directly. This gives full control over the detection rules. "
    "Accounts without a specific override use the global default.")

st.markdown("---")


# -- Governance stance --------------------------------------------------------
st.markdown("#### What the output means")
st.write(
    "The flags and the AI's assessment are prompts for a human to review, "
    "not verdicts. A flagged account means the tool found something "
    "unusual enough to warrant a look. The AI's triage (priority, "
    "timing versus permanent, likely cause) is a hypothesis to help you "
    "focus, not a conclusion. The controller decides what needs action.")

st.write(
    "Every run records the data hashes (so the exact input is provable), "
    "the effective thresholds (so the sensitivity is on record), and the "
    "token cost. The review PDF and the anomalies CSV are available for "
    "download after triage, giving the controller a complete artefact to "
    "file or circulate.")

st.divider()
st.caption("Sample data only. All figures are illustrative.")
