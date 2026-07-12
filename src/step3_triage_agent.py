# =============================================================================
# step3_triage_agent.py — Layer 4: AI Triage Agent
# =============================================================================
# Responsibilities:
#   - build_prompt():        construct the triage prompt
#   - call_claude():         send prompts, return the response
#   - parse_triage_json():   extract structured triage data from the response
#   - strip_json_block():    remove the JSON block so the narrative is clean
#
# The agent triages the anomalies Python detected: ranks them, judges timing
# versus permanent, hypothesises a cause. It returns BOTH a narrative (for
# the human to read) AND a structured JSON block (for the CSV artefact).
#
# Hard rule enforced in the prompt: the agent must return EVERY flagged
# account and may never drop one.
# =============================================================================

import re
import json
import anthropic

from config import (
    ANTHROPIC_API_KEY, MODEL, MAX_TOKENS,
    BENCHMARKS, BENCHMARK_MEANING,
)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def build_prompt(flagged, entity, close_period):
    """
    Build the system and user prompts. Claude receives the detected
    anomalies with full context and must triage them, returning a narrative
    plus a structured JSON block. It must never drop a flag.
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
        "- Keep each hypothesis to one concise clause, under 15 words. "
        "State the likely cause plainly, no hedging language.\n"
        "- In the JSON, the account field must copy the account name EXACTLY "
        "as given in the flagged anomalies, character for character, including "
        "any ampersand or abbreviation. Do not reword, expand, or normalise it.\n"
        "- Use only standard ASCII characters. No arrows, em dashes, or en "
        "dashes. Use plain words and commas or full stops.\n"
        "</hard_rules>\n\n"
        "<output_format>\n"
        "First, write the narrative. Start with one sentence stating how many "
        "accounts need attention and how many are likely timing versus "
        "permanent. Then, ordered by priority (HIGH first), for each account:\n\n"
        "ACCOUNT NAME [PRIORITY]\n"
        "Likely: timing or permanent, with a one line reason.\n"
        "Story: which benchmark matters and what it says.\n"
        "Hypothesis: a plain-English guess at the cause to verify.\n"
        "Confidence: high, medium, or low.\n\n"
        "End the narrative with a RECOMMENDED ACTIONS line naming the one or "
        "two accounts to investigate first.\n\n"
        "Then, after the narrative, output a JSON block in exactly this "
        "format, inside a fenced code block marked json. Include EVERY "
        "flagged account:\n"
        "```json\n"
        "{{\n"
        '  "triage": [\n'
        '    {{"account": "NAME", "priority": "HIGH|MEDIUM|LOW", '
        '"assessment": "timing|permanent", '
        '"headline": "the single most important number, e.g. 46% over budget (EUR 45k)", '
        '"story": "which benchmark tells the real story and what it means, one clause", '
        '"hypothesis": "likely cause, under 15 words", '
        '"confidence": "high|medium|low"}}\n'
        "  ]\n"
        "}}\n"
        "```\n"
        "</output_format>"
    ).format(entity=entity)

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
        "Triage every flagged account in the exact output format specified, "
        "including the JSON block."
    ).format(
        entity=entity, period=close_period,
        n=len(flagged), anomalies=anomaly_block,
    )

    print("\n[OK] Prompt built")
    print("     Flagged accounts in prompt: {}".format(len(flagged)))
    return system_prompt, user_prompt


def parse_triage_json(response_text):
    """
    Extract the structured triage list from the response. Looks for a
    fenced json block first, then a bare object as a fallback. Returns the
    list of triage dicts, or None if not found or invalid, so the pipeline
    degrades gracefully rather than crashing.
    """
    match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if not match:
        match = re.search(r'(\{.*"triage".*\})', response_text, re.DOTALL)
        if not match:
            return None
    raw = match.group(1)
    try:
        data = json.loads(raw)
        return data.get("triage", None)
    except json.JSONDecodeError:
        return None


def strip_json_block(response_text):
    """Remove the fenced json block so the narrative is clean for text/PDF."""
    return re.sub(r'```json\s*.*?\s*```', '', response_text, flags=re.DOTALL).strip()


def call_claude(system_prompt, user_prompt):
    """Send prompts to Claude and return the response plus token counts."""
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

    return text, input_tokens, output_tokens, stop_reason
