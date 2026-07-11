# =============================================================================
# main.py — Anomaly Detection and Alert Agent
# Pass 1: flat script, console output, understand every line
# =============================================================================
#
# Python DETECTS the statistical anomalies. Claude TRIAGES them: ranks,
# judges timing versus permanent, and hypothesises a cause. The agent
# never suppresses a flag; every Python-flagged account stays on the list.
# =============================================================================

import pandas as pd
import anthropic
import json
import hashlib
import statistics
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

from config import (
    ANTHROPIC_API_KEY, MODEL, MAX_TOKENS,
    CLOSE_FILE, HISTORY_FILE, OUTPUT_DIR, AUDIT_LOG,
    DEFAULT_ENTITY, CLOSE_PERIOD,
    BENCHMARKS, BENCHMARK_MEANING, NEAR_ZERO,
    GLOBAL_THRESHOLD, ACCOUNT_THRESHOLDS, MODIFIED_Z_CUTOFF,
)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# =============================================================================
# STEP 1: Load the close data and the history
# =============================================================================
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

    required = {"account", "actual", "budget", "prior", "forecast"}
    missing  = required - set(df.columns)
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
    Load 12+ months of trailing history per account. Returns a dict of
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

    required = {"account", "period", "value"}
    missing  = required - set(df.columns)
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


# =============================================================================
# STEP 2: Detection helpers
# =============================================================================
def thresholds_for(account):
    """Return the threshold config for an account, or the global default."""
    return ACCOUNT_THRESHOLDS.get(account, GLOBAL_THRESHOLD)


def safe_variance(actual, benchmark):
    """
    Compute the dollar change always. Compute the percentage only if the
    benchmark is above the near-zero floor. Near-zero returns pct=None so
    we never divide by a tiny base and never produce an absurd percentage.
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
    Robust to small samples and outliers because it uses the median.

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


# =============================================================================
# STEP 3: Run detection across all accounts
# =============================================================================
def detect_anomalies(close_df, history):
    """
    For each account, compute variance against all three benchmarks, apply
    the materiality rule, and add the volatility context. An account is
    flagged if it is material against any benchmark OR its history is flat
    and it moved. Returns a list of flagged account dicts plus flags.
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

        comparisons          = {}
        material_benchmarks  = []
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


# =============================================================================
# STEP 4: Build the triage prompt
# =============================================================================
def build_prompt(flagged, entity, close_period):
    """
    Build the system and user prompts. Claude receives the detected
    anomalies with full context and must triage them: rank, judge timing
    versus permanent, hypothesise a cause. It must NEVER drop a flag.
    """
    system_prompt = (
        "You are a financial controller reviewing the month-end close for "
        "{entity}. A detection system has already scanned every account and "
        "flagged the ones below as statistical anomalies. Your job is to "
        "triage them for the finance team.\n\n"
        "<what_the_detection_did>\n"
        "Each account was compared against three benchmarks:\n"
        "- prior: last month actual (a flux and error check)\n"
        "- budget: the plan (a performance check)\n"
        "- forecast: the latest expectation (a drift check)\n"
        "An account was flagged if a variance was material (large in "
        "percentage and euro terms) or if a normally stable account moved.\n"
        "</what_the_detection_did>\n\n"
        "<your_job>\n"
        "For each flagged account:\n"
        "1. Judge whether the movement is most likely TIMING (a one-off that "
        "reverses next month, such as an annual invoice, a prepayment, a "
        "campaign, a conference) or PERMANENT (a real step change).\n"
        "2. Say which benchmark tells the real story. An account off budget "
        "but on forecast is a very different situation from off both.\n"
        "3. Give a one line plain-English hypothesis for the cause.\n"
        "4. Assign a priority: HIGH, MEDIUM, or LOW for the controller.\n"
        "5. Note your confidence and flag anything you are unsure about.\n"
        "</your_job>\n\n"
        "<hard_rules>\n"
        "- You must return EVERY flagged account. Never drop or hide one, "
        "even if you think it is a false positive. If you believe it is "
        "noise, keep it on the list and mark it LOW with your reasoning.\n"
        "- Never invent numbers. Use only the figures provided.\n"
        "- Your hypotheses are suggestions for a human to verify, not "
        "conclusions. Phrase them as such.\n"
        "- Use only standard ASCII characters. No arrows, em dashes, or en "
        "dashes. Use plain words and commas or full stops.\n"
        "</hard_rules>\n\n"
        "<output_format>\n"
        "Start with one sentence stating how many accounts need attention "
        "and how many are likely timing versus permanent.\n\n"
        "Then, ordered by priority (HIGH first), for each account:\n\n"
        "ACCOUNT NAME [PRIORITY]\n"
        "Likely: timing or permanent, with a one line reason.\n"
        "Story: which benchmark matters and what it says.\n"
        "Hypothesis: a plain-English guess at the cause to verify.\n"
        "Confidence: high, medium, or low.\n\n"
        "End with a RECOMMENDED ACTIONS line naming the one or two accounts "
        "the controller should investigate first.\n"
        "</output_format>"
    ).format(entity=entity)

    # Build the anomaly context block
    lines = []
    for f in flagged:
        lines.append("ACCOUNT: {} (actual EUR {:,.0f})".format(f["account"], f["actual"]))
        for bench in BENCHMARKS:
            c = f["comparisons"][bench]
            if c["pct"] is not None:
                pct_str = "{:+.1%}".format(c["pct"])
            else:
                pct_str = "n/a (near-zero base)"
            mark = "  [MATERIAL: {}]".format(c["reason"]) if c["material"] else ""
            lines.append("  vs {} ({}): {:+,.0f}, {}{}".format(
                bench, BENCHMARK_MEANING[bench], c["dollar"], pct_str, mark))
        vol = f["volatility"]
        if isinstance(f["modified_z"], float):
            vol += " (modified z-score {:+.2f})".format(f["modified_z"])
        lines.append("  volatility context: {}".format(vol))
        lines.append("")
    anomaly_block = "\n".join(lines)

    user_prompt = (
        "CLOSE REVIEW\n"
        "Entity: {entity}\n"
        "Period: {period}\n"
        "Accounts flagged for triage: {n}\n\n"
        "<flagged_anomalies>\n"
        "{anomalies}\n"
        "</flagged_anomalies>\n\n"
        "Triage every flagged account in the exact output format specified."
    ).format(
        entity=entity, period=close_period,
        n=len(flagged), anomalies=anomaly_block,
    )

    print("\n[OK] Prompt built")
    print("     Flagged accounts in prompt: {}".format(len(flagged)))
    return system_prompt, user_prompt


# =============================================================================
# STEP 5: Call Claude
# =============================================================================
def call_claude(system_prompt, user_prompt):
    print("\n[..] Calling Claude API ({})...".format(MODEL))
    try:
        response = client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except anthropic.AuthenticationError:
        raise RuntimeError("Authentication failed. Check ANTHROPIC_API_KEY in .env.")
    except anthropic.RateLimitError:
        raise RuntimeError("Rate limit reached. Wait 60 seconds and try again.")
    except anthropic.APIStatusError as e:
        raise RuntimeError("API error {}: {}".format(e.status_code, e.message))
    except anthropic.APIConnectionError:
        raise RuntimeError("Cannot connect to Anthropic API. Check connection.")

    if not response.content:
        raise RuntimeError("Claude returned an empty response.")

    text          = response.content[0].text
    input_tokens  = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    stop_reason   = response.stop_reason

    if stop_reason == "max_tokens":
        print("[WARN] Response truncated. Consider increasing MAX_TOKENS.")
    if output_tokens < 150 and stop_reason != "max_tokens":
        print("[WARN] Unusually short output ({} tokens).".format(output_tokens))

    cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)

    print("[OK] Claude responded")
    print("     Stop reason:   {}".format(stop_reason))
    print("     Tokens:        {:,} in / {:,} out".format(input_tokens, output_tokens))
    print("     Approx cost:   EUR {:.4f}".format(cost))
    print("\n" + "=" * 64)
    print("CLOSE ANOMALY TRIAGE — {}".format(DEFAULT_ENTITY))
    print("=" * 64)
    print(text)
    print("=" * 64)

    return text, input_tokens, output_tokens, stop_reason


# =============================================================================
# STEP 6: Write output and audit log
# =============================================================================
def write_output(triage, flagged, flags, tok_in, tok_out, stop_reason):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    now     = datetime.now(timezone.utc)
    ts_file = now.strftime("%Y-%m-%d_%H-%M-%S")
    ts_log  = now.isoformat()

    out_path = OUTPUT_DIR / "close_triage_{}.txt".format(ts_file)
    header = (
        "CLOSE ANOMALY TRIAGE - GENERATED OUTPUT\n"
        "{sep}\n"
        "Generated:  {ts}\n"
        "Entity:     {entity}\n"
        "Period:     {period}\n"
        "Flagged:    {n} accounts\n"
        "Model:      {model}\n"
        "Tokens:     {ti:,} in / {to:,} out\n"
        "{sep}\n\n"
    ).format(
        sep="=" * 64, ts=ts_log, entity=DEFAULT_ENTITY, period=CLOSE_PERIOD,
        n=len(flagged), model=MODEL, ti=tok_in, to=tok_out,
    )
    out_path.write_text(header + triage, encoding="utf-8")

    with open(CLOSE_FILE, "rb") as fh:
        close_hash = "sha256:" + hashlib.sha256(fh.read()).hexdigest()
    with open(HISTORY_FILE, "rb") as fh:
        history_hash = "sha256:" + hashlib.sha256(fh.read()).hexdigest()

    requires_review = (
        len(flagged) > 0
        or len(flags) > 0
        or stop_reason == "max_tokens"
        or (tok_out < 150 and stop_reason != "max_tokens")
    )

    audit = {
        "run_id":          ts_log,
        "project":         "anomaly-detection-alert-agent",
        "entity":          DEFAULT_ENTITY,
        "close_period":    CLOSE_PERIOD,
        "accounts_flagged": len(flagged),
        "flagged_accounts": [f["account"] for f in flagged],
        "close_hash":      close_hash,
        "history_hash":    history_hash,
        "output_file":     str(out_path),
        "model":           MODEL,
        "input_tokens":    tok_in,
        "output_tokens":   tok_out,
        "stop_reason":     stop_reason,
        "detection_flags": flags,
        "human_reviewed":  False,
        "requires_review": requires_review,
    }
    with open(AUDIT_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(audit) + "\n")

    print("\n[OK] Output written")
    print("     Triage:    {}".format(out_path.name))
    print("     Audit log: {}".format(AUDIT_LOG.name))
    print("     Close hash:   {}...".format(close_hash[:30]))
    print("     History hash: {}...".format(history_hash[:30]))
    print("     Requires human review: {}".format(requires_review))
    return out_path, audit


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    close_df = load_close(CLOSE_FILE)
    history  = load_history(HISTORY_FILE)

    flagged, flags = detect_anomalies(close_df, history)

    if not flagged:
        print("\n[DONE] No anomalies flagged. Clean close.")
    else:
        system_prompt, user_prompt = build_prompt(flagged, DEFAULT_ENTITY, CLOSE_PERIOD)
        triage, tok_in, tok_out, stop_reason = call_claude(system_prompt, user_prompt)
        out_path, audit = write_output(triage, flagged, flags, tok_in, tok_out, stop_reason)

        print("\n[DONE] Triage complete.")
        print("       Output: {}".format(out_path.name))
        if audit["requires_review"]:
            print("\n" + "!" * 64)
            print("  HUMAN REVIEW REQUIRED — {} account(s) flagged".format(len(flagged)))
            print("!" * 64)
