# =============================================================================
# lib/sensitivity.py — slider-to-threshold mapping for the web layer
# =============================================================================
# Two plain-language sensitivity levels (materiality and volatility) map to
# the real threshold values the detection engine uses. The sliders scale the
# config baselines proportionally, preserving the per-account structure.
#
# The advanced expander can patch individual values on top. Whatever the
# effective set holds is what detection uses.
#
# Pure functions only: no Streamlit, no IO, so they import and test cleanly.
# =============================================================================

from config import (
    GLOBAL_THRESHOLD, ACCOUNT_THRESHOLDS,
    MODIFIED_Z_CUTOFF, NEAR_ZERO,
)

# ── Sensitivity levels ──────────────────────────────────────────────────────
# Each level names a materiality scale factor and a z-score cutoff. Higher
# sensitivity means lower thresholds (more flags). The factor multiplies the
# materiality values; a factor of 0.5 halves the bands (twice as sensitive).
#
# The levels are chosen so that each step visibly changes the flagged set on
# the sample data.

LEVELS = ["Very low", "Low", "Medium", "High", "Very high"]
BASELINE_INDEX = 2  # "Medium" is the config default

MATERIALITY_FACTORS = {
    "Very low":  3.0,
    "Low":       2.0,
    "Medium":    1.0,
    "High":      0.5,
    "Very high": 0.25,
}

Z_CUTOFFS = {
    "Very low":  8.0,
    "Low":       5.5,
    "Medium":    MODIFIED_Z_CUTOFF,  # 3.5
    "High":      2.5,
    "Very high": 2.0,
}


def scale_threshold(base, factor):
    """Scale a single materiality threshold dict by a factor."""
    return {
        "pct_band":         base["pct_band"] * factor,
        "min_dollar_floor": base["min_dollar_floor"] * factor,
        "big_dollar":       base["big_dollar"] * factor,
    }


def effective_thresholds(materiality_level, volatility_level):
    """Compute the effective threshold set from two slider levels.

    Returns (account_thresholds, global_threshold, z_cutoff, near_zero),
    ready to pass straight into detect_anomalies().
    """
    mat_factor = MATERIALITY_FACTORS[materiality_level]
    z_cutoff   = Z_CUTOFFS[volatility_level]

    scaled_global = scale_threshold(GLOBAL_THRESHOLD, mat_factor)
    scaled_accounts = {
        account: scale_threshold(base, mat_factor)
        for account, base in ACCOUNT_THRESHOLDS.items()
    }

    return scaled_accounts, scaled_global, z_cutoff, NEAR_ZERO


def patch_thresholds(account_thresholds, global_threshold, z_cutoff, near_zero,
                     patches):
    """Apply user overrides from the advanced expander on top of the
    slider-derived effective set.

    patches is a dict of:
      {"global": {field: value, ...},
       "z_cutoff": float,
       "near_zero": float,
       "accounts": {account: {field: value, ...}, ...}}

    Returns the patched (account_thresholds, global_threshold, z_cutoff,
    near_zero) tuple.
    """
    gt = dict(global_threshold)
    at = {k: dict(v) for k, v in account_thresholds.items()}
    zc = z_cutoff
    nz = near_zero

    if "global" in patches:
        gt.update(patches["global"])
    if "z_cutoff" in patches:
        zc = patches["z_cutoff"]
    if "near_zero" in patches:
        nz = patches["near_zero"]
    if "accounts" in patches:
        for account, overrides in patches["accounts"].items():
            if account in at:
                at[account].update(overrides)
            else:
                at[account] = dict(gt)
                at[account].update(overrides)

    return at, gt, zc, nz
