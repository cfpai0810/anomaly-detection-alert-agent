# =============================================================================
# config.py — Anomaly Detection and Alert Agent
# =============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found. Check .env in project root.")

MODEL      = "claude-sonnet-4-6"
MAX_TOKENS = 2048

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
