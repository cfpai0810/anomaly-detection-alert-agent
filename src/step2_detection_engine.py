# =============================================================================
# step2_detection_engine.py — Layer 3: Anomaly Detection
# =============================================================================
# Responsibilities:
#   - thresholds_for():      per-account threshold lookup
#   - safe_variance():       dollar always, percent only if base not near-zero
#   - is_material():         three-part materiality rule
#   - volatility_context():  modified z-score as a context signal
#   - detect_anomalies():    run all checks across every account
#
# Knows about: the detection maths, thresholds, statistics
# Does NOT know about: file loading, Claude, output files
#
# Core rule: this layer is fully deterministic. Same inputs always produce
# the same flags. Nothing here calls the language model.
# =============================================================================

import statistics

from config import (
    BENCHMARKS, NEAR_ZERO,
    GLOBAL_THRESHOLD, ACCOUNT_THRESHOLDS, MODIFIED_Z_CUTOFF,
)


def thresholds_for(account):
    """Return the threshold config for an account, or the global default."""
    return ACCOUNT_THRESHOLDS.get(account, GLOBAL_THRESHOLD)


def safe_variance(actual, benchmark):
    """
    Compute the dollar change always. Compute the percentage only if the
    benchmark is above the near-zero floor, so we never divide by a tiny
    base and never produce an absurd percentage. Returns (dollar, pct, base)
    where pct is None for a near-zero base.
    """
    dollar = actual - benchmark
    if abs(benchmark) < NEAR_ZERO:
        return dollar, None, "near_zero_base"
    return dollar, dollar / benchmark, "ok"


def is_material(account, dollar, pct):
    """
    Three-part materiality rule:
      (|pct| >= pct_band AND |dollar| >= min_dollar_floor)
        OR (|dollar| >= big_dollar)
    If pct is None (near-zero base), only the big-dollar test applies.
    Returns (is_material, reason).
    """
    t = thresholds_for(account)
    big = abs(dollar) >= t["big_dollar"]

    if pct is None:
        return (big, "big_dollar" if big else None)

    pct_and_floor = (abs(pct) >= t["pct_band"]) and (abs(dollar) >= t["min_dollar_floor"])
    if pct_and_floor and big:
        return (True, "pct+floor & big_dollar")
    if pct_and_floor:
        return (True, "pct+floor")
    if big:
        return (True, "big_dollar")
    return (False, None)


def volatility_context(value, history_values):
    """
    Modified z-score (median and MAD) as a CONTEXT signal, not a hard flag.
    Robust to small samples and outliers because it uses the median, which
    is not dragged around by the very anomalies we are trying to detect.

    Returns (label, modified_z):
      unusual_for_account    z beyond the cutoff, large for this account
      within_normal_range    fits the account's usual volatility
      stable_account_moved   flat history (MAD=0) but the value moved
      stable_no_move         flat history and no movement
    """
    med = statistics.median(history_values)
    mad = statistics.median([abs(x - med) for x in history_values])

    if mad == 0:
        if value != med:
            return ("stable_account_moved", None)
        return ("stable_no_move", 0.0)

    mz = 0.6745 * (value - med) / mad
    if abs(mz) > MODIFIED_Z_CUTOFF:
        return ("unusual_for_account", mz)
    return ("within_normal_range", mz)


def detect_anomalies(close_df, history):
    """
    For each account, compute variance against all three benchmarks, apply
    the materiality rule, and add the volatility context. An account is
    flagged if it is material against any benchmark OR its history is flat
    and it moved. Returns (flagged list, detection flags list).
    """
    flagged = []
    flags   = []

    for _, row in close_df.iterrows():
        account = row["account"]
        actual  = row["actual"]

        if account not in history:
            flags.append("MISSING_HISTORY: {} has no trailing history".format(account))
            history_values = [actual]
        else:
            history_values = history[account]

        comparisons         = {}
        material_benchmarks = []
        for bench in BENCHMARKS:
            dollar, pct, base = safe_variance(actual, row[bench])
            material, reason  = is_material(account, dollar, pct)
            comparisons[bench] = {
                "benchmark_value": row[bench],
                "dollar": dollar, "pct": pct, "base": base,
                "material": material, "reason": reason,
            }
            if material:
                material_benchmarks.append(bench)

        vol_label, mz = volatility_context(actual, history_values)
        stable_moved  = (vol_label == "stable_account_moved")

        is_flagged = bool(material_benchmarks) or stable_moved
        if is_flagged:
            flagged.append({
                "account": account,
                "actual": actual,
                "comparisons": comparisons,
                "material_benchmarks": material_benchmarks,
                "volatility": vol_label,
                "modified_z": mz,
                "stable_moved": stable_moved,
            })

    print("\n[OK] Detection complete")
    print("     Accounts scanned: {}".format(len(close_df)))
    print("     Flagged:          {}".format(len(flagged)))
    print("     Flags raised:     {}".format(len(flags)))
    for f in flagged:
        reasons = []
        if f["material_benchmarks"]:
            reasons.append("material vs " + "+".join(f["material_benchmarks"]))
        if f["stable_moved"]:
            reasons.append("stable account moved")
        print("     --> {}: {}".format(f["account"], "; ".join(reasons)))

    return flagged, flags
