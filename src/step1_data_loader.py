# =============================================================================
# step1_data_loader.py — Layer 2: Data Loading and Validation
# =============================================================================
# Responsibilities:
#   - load_close():    load the month-end close with three benchmarks
#   - load_history():  load trailing history per account
#
# Knows about: pandas, file paths, validation
# Does NOT know about: detection maths, Claude, output files
# =============================================================================

import pandas as pd
from pathlib import Path

from config import CLOSE_PERIOD

REQUIRED_CLOSE_COLS   = {"account", "actual", "budget", "prior", "forecast"}
REQUIRED_HISTORY_COLS = {"account", "period", "value"}


def load_close(filepath):
    """
    Load the month-end close: one row per account with actual and three
    benchmarks (prior month, budget, forecast).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError("Close file not found: {}".format(filepath))

    df = pd.read_csv(filepath, dtype={
        "account": "str", "actual": "float64",
        "budget": "float64", "prior": "float64", "forecast": "float64",
    })
    df["account"] = df["account"].str.strip()

    missing = REQUIRED_CLOSE_COLS - set(df.columns)
    if missing:
        raise ValueError("Close CSV missing columns: {}".format(sorted(missing)))

    if df["account"].duplicated().any():
        dupes = df[df["account"].duplicated()]["account"].tolist()
        raise ValueError("Duplicate accounts in close: {}".format(dupes))

    print("[OK] Close loaded")
    print("     Accounts: {}".format(len(df)))
    print("     Period:   {}".format(CLOSE_PERIOD))
    return df


def load_history(filepath):
    """
    Load trailing history per account. Returns a dict of
    {account: [values oldest to newest]} for the statistical layer.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError("History file not found: {}".format(filepath))

    df = pd.read_csv(filepath, dtype={
        "account": "str", "period": "str", "value": "float64",
    })
    df["account"] = df["account"].str.strip()
    df["period"]  = df["period"].str.strip()

    missing = REQUIRED_HISTORY_COLS - set(df.columns)
    if missing:
        raise ValueError("History CSV missing columns: {}".format(sorted(missing)))

    history = {}
    for account, group in df.groupby("account"):
        ordered = group.sort_values("period")
        history[account] = ordered["value"].tolist()

    months = len(next(iter(history.values()))) if history else 0
    print("[OK] History loaded")
    print("     Accounts: {}".format(len(history)))
    print("     Months per account: {}".format(months))
    return history
