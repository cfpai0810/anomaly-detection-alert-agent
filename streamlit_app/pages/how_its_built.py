# =============================================================================
# pages/how_its_built.py -- the technical and governance story
# =============================================================================

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit_app.lib.theme import inject_css
from streamlit_app.lib.governance_lifecycle import governance_lifecycle_svg

st.markdown(inject_css(), unsafe_allow_html=True)


# -- Title --------------------------------------------------------------------
st.markdown("### How it's built")
st.write(
    "The technical design and the governance properties that make the tool "
    "trustworthy. This page is for anyone who wants to understand what "
    "runs where, what the AI can and cannot do, and how the audit trail "
    "works.")

st.markdown("---")


# -- The boundary -------------------------------------------------------------
st.markdown("#### The boundary: Python detects, the model triages")
st.write(
    "This is the heart of the design. Python does all the detection: "
    "the variance computation, the materiality rule, the modified z-score, "
    "the stable-moved check, and the flagging decision. The detection "
    "engine is a pure function with no randomness and no model dependency. "
    "The same inputs and the same thresholds always produce the same "
    "flagged accounts.")
st.write(
    "The model (Claude) only triages accounts that Python has already "
    "flagged. It receives the variance data Python computed and returns a "
    "priority, an assessment (timing or permanent), a hypothesis, and a "
    "confidence level. It never sees the unflagged accounts, never decides "
    "what is suspicious, and cannot remove a flag or suppress an account "
    "from the review.")

st.info(
    "The AI adds interpretation on top of a deterministic base. It does "
    "not control what is flagged, and it cannot hide what the detection "
    "surfaces.")

st.markdown("---")


# -- The pipeline -------------------------------------------------------------
st.markdown("#### The pipeline")

_PIPELINE_SVG = """
<svg viewBox="0 0 720 120" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:720px;font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <!-- boxes -->
  <rect x="2" y="30" width="110" height="60" rx="6" fill="#EAF2FB" stroke="#2D6A9F" stroke-width="1.5"/>
  <rect x="152" y="30" width="110" height="60" rx="6" fill="#EAF2FB" stroke="#2D6A9F" stroke-width="1.5"/>
  <rect x="302" y="30" width="110" height="60" rx="6" fill="#FAEEDA" stroke="#854F0B" stroke-width="1.5"/>
  <rect x="452" y="30" width="110" height="60" rx="6" fill="#EAF3DE" stroke="#1D6B0F" stroke-width="1.5"/>
  <rect x="602" y="30" width="110" height="60" rx="6" fill="#EAF2FB" stroke="#2D6A9F" stroke-width="1.5"/>
  <!-- labels -->
  <text x="57" y="55" text-anchor="middle" fill="#1A3A5C" font-size="11" font-weight="bold">Load</text>
  <text x="57" y="72" text-anchor="middle" fill="#898781" font-size="9">data files</text>
  <text x="207" y="55" text-anchor="middle" fill="#1A3A5C" font-size="11" font-weight="bold">Detect</text>
  <text x="207" y="72" text-anchor="middle" fill="#898781" font-size="9">Python, deterministic</text>
  <text x="357" y="55" text-anchor="middle" fill="#854F0B" font-size="11" font-weight="bold">Review flags</text>
  <text x="357" y="72" text-anchor="middle" fill="#898781" font-size="9">human decision</text>
  <text x="507" y="55" text-anchor="middle" fill="#1D6B0F" font-size="11" font-weight="bold">Triage</text>
  <text x="507" y="72" text-anchor="middle" fill="#898781" font-size="9">AI, on request</text>
  <text x="657" y="55" text-anchor="middle" fill="#1A3A5C" font-size="11" font-weight="bold">Output</text>
  <text x="657" y="72" text-anchor="middle" fill="#898781" font-size="9">PDF, CSV, audit</text>
  <!-- arrows -->
  <line x1="114" y1="60" x2="148" y2="60" stroke="#2D6A9F" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="264" y1="60" x2="298" y2="60" stroke="#2D6A9F" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="414" y1="60" x2="448" y2="60" stroke="#854F0B" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="564" y1="60" x2="598" y2="60" stroke="#1D6B0F" stroke-width="1.5" marker-end="url(#arr)"/>
  <!-- top labels -->
  <text x="207" y="18" text-anchor="middle" fill="#2D6A9F" font-size="9">free, instant</text>
  <text x="357" y="18" text-anchor="middle" fill="#854F0B" font-size="9">you decide to proceed</text>
  <text x="507" y="18" text-anchor="middle" fill="#1D6B0F" font-size="9">costs tokens</text>
  <defs><marker id="arr" viewBox="0 0 10 10" refX="10" refY="5"
    markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#2D6A9F"/></marker></defs>
</svg>
"""

components.html(_PIPELINE_SVG, height=130, scrolling=False)
st.caption(
    "The human gate between detection and triage is deliberate: the "
    "controller reviews the flags before choosing to invoke the AI.")

st.write(
    "The pipeline has four steps. **Load** reads the close actuals and "
    "the trailing account history from CSV files. **Detect** runs the "
    "materiality and volatility tests against the effective thresholds "
    "and produces the flagged set. The controller reviews the flags "
    "(adjusting sensitivity if needed) and decides whether to proceed. "
    "**Triage** sends the flagged accounts to Claude for interpretation. "
    "**Output** produces the review PDF, the anomalies CSV with a "
    "disposition column, and the audit record.")

st.markdown("---")


# -- Determinism and lineage -------------------------------------------------
st.markdown("#### Determinism and lineage")
st.write(
    "The detection engine is a pure function: it takes the close data, "
    "the history, the threshold parameters, the z-score cutoff, and the "
    "near-zero floor, and returns the flagged accounts. No randomness, "
    "no network calls, no side effects. The same inputs and the same "
    "thresholds always produce the same flags.")
st.write(
    "Each run records a governance trail:")

st.write(
    "- **Data hashes.** SHA-256 hashes of the close and history files, "
    "so the exact input is provable after the fact.\n"
    "- **Effective thresholds.** The materiality thresholds and volatility "
    "cutoff actually used, whether from the sensitivity sliders or set "
    "directly as custom thresholds.\n"
    "- **Tokens and cost.** The input and output token counts and an "
    "estimated cost.\n"
    "- **Requires-review flag.** Set when accounts are flagged, when "
    "the model's response was truncated, or when the output is "
    "suspiciously short.")

st.markdown("---")


# -- The every-account rule ---------------------------------------------------
st.markdown("#### The every-account rule")
st.write(
    "The model is asked to return an assessment for every flagged account. "
    "After the response arrives, the tool matches the model's output "
    "against the flagged set by normalised account name. If any flagged "
    "account is missing from the response, the tool surfaces it with a "
    "warning rather than hiding it. A missing account is always visible "
    "to the reviewer.")

st.markdown("---")


# -- The materiality rule in detail -------------------------------------------
st.markdown("#### Detection rules in detail")

st.markdown("**The three-part materiality rule.**")
st.write(
    "For each benchmark (prior, budget, forecast), the variance is "
    "material if: (a) the percentage clears the account's band AND the "
    "currency amount clears the floor, OR (b) the currency amount alone clears "
    "the \"big move\" threshold. Each account type has its own thresholds "
    "(Revenue and Personnel are tighter; discretionary costs are looser), "
    "with a global default for accounts without a specific override. The "
    "sensitivity slider scales all thresholds by a single factor.")

st.markdown("**The modified z-score.**")
st.write(
    "The tool uses the Iglewicz-Hoaglin modified z-score, which compares "
    "the current value to the median of the trailing twelve months, "
    "scaled by the median absolute deviation (MAD). This is robust to "
    "outliers in the history: one extreme month does not drag the "
    "baseline the way a simple mean would. The volatility slider sets "
    "the cutoff directly.")

st.markdown("**Near-zero guard.**")
st.write(
    "When a benchmark value is below the near-zero floor, no percentage "
    "is computed. This prevents a tiny or zero base from producing a "
    "misleading large percentage. The dollar variance is still computed "
    "and can still trigger the big-move rule.")

st.markdown("---")


# -- Governance stance --------------------------------------------------------
st.markdown("#### Flag-for-review governance")
st.write(
    "The AI's output is explicitly framed as hypotheses for a human to "
    "verify. The model's system prompt instructs it to assess every "
    "flagged account, to state its confidence, and to avoid presenting "
    "its judgement as fact. The review PDF, the anomalies CSV, and the "
    "session audit trail are designed so the controller has a complete "
    "record of what was flagged, at what sensitivity, from which data, "
    "and what the model said, before signing off on any disposition.")

st.write(
    "The tool is an aid to the review, not a replacement for it. The "
    "controller decides what needs action.")

st.markdown("#### The governance lifecycle")
components.html(governance_lifecycle_svg(), height=310, scrolling=False)
st.write(
    "In a production deployment, review is the start of the record, not "
    "the end. Once a reviewer approves the triage, the approved version "
    "becomes the immutable record: it is what everyone reads afterwards, "
    "and it cannot be edited. A later change is issued as a new, "
    "separately approved version, and the original approved triage still "
    "stands unchanged. This live demo stops at flagging for review; it "
    "does not store or lock anything, so no data is retained. The "
    "lifecycle diagram above shows the full model this design is built "
    "towards.")

st.divider()
st.caption("Sample data only. All figures are illustrative.")
