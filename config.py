# =============================================================================
# config.py — Anomaly Detection and Alert Agent
# =============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Optional at import time. The CLI validates it when building its client
# (see src/step3); the web supplies a per-session key via the key gate.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

MODEL      = "claude-sonnet-4-6"
MAX_TOKENS = 2048

# Cost-estimate rates. These are an ESTIMATE input, not a bill; Anthropic's
# metering is authoritative.
#
# Model: claude-sonnet-4-6
# Rate:  $3.00 per million input tokens, $15.00 per million output tokens.
# Verified: August 2026. Re-verify at https://www.anthropic.com/pricing
# IMPORTANT: this rate is tied to the model above. If MODEL changes (for example
# to a newer Sonnet), update these rates to match, or the estimate will be wrong.
COST_INPUT_USD_PER_MTOK  = 3.00
COST_OUTPUT_USD_PER_MTOK = 15.00

# Assumed EUR/USD for display. The API bills in USD; this is a fixed assumption
# for the euro figure, clearly labelled as such wherever shown.
ASSUMED_USD_PER_EUR = 1.08

# The currency the tool DISPLAYS data figures in. Separate from the API cost
# estimate, which reports the Anthropic bill in EUR and USD.
CURRENCY_CODE = "EUR"
CURRENCY_SYMBOL = "€"

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

CLOSE_FILE   = DATA_DIR / "close_actuals.csv"
HISTORY_FILE = DATA_DIR / "account_history.csv"
AUDIT_LOG    = OUTPUT_DIR / "audit_log.jsonl"

DEFAULT_ENTITY = "Valencia Operations"
CLOSE_PERIOD   = "2026-06"

# Benchmarks and what each one means (fed to the triage agent)
BENCHMARKS = ["prior", "budget", "forecast"]
BENCHMARK_MEANING = {
    "prior":    "flux and error check versus last month actual",
    "budget":   "performance versus plan",
    "forecast": "drift versus latest expectation",
}

# A benchmark below this is treated as near-zero: no percentage computed
NEAR_ZERO = 1000

# Three-part materiality thresholds. Per-account overrides a global default.
GLOBAL_THRESHOLD = {"pct_band": 0.10, "min_dollar_floor": 10000, "big_dollar": 50000}
ACCOUNT_THRESHOLDS = {
    "Revenue":        {"pct_band": 0.05, "min_dollar_floor": 20000, "big_dollar": 75000},
    "Personnel Cost": {"pct_band": 0.05, "min_dollar_floor": 15000, "big_dollar": 50000},
    "COGS":           {"pct_band": 0.08, "min_dollar_floor": 15000, "big_dollar": 60000},
}

# Modified z-score cutoff (Iglewicz-Hoaglin standard)
MODIFIED_Z_CUTOFF = 3.5
