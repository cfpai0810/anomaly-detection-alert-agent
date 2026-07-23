# Anomaly Detection and Alert Agent

A demonstration of how AI can support the month-end close: Python scans every
account for statistical anomalies, and an AI agent triages them for the
controller, judging which are worth attention and why. The controller stays
in control; the agent makes the review faster, not automatic.

---

## What it does

Takes a month-end close (each account with its actual and three benchmarks)
plus each account's trailing history, then:

1. **Detects** anomalies deterministically in Python: material variances
   against prior month, budget, and forecast, plus statistical outliers versus
   each account's own history.
2. **Triages** them with the Claude API: the agent ranks each flagged account,
   judges whether the movement is timing or permanent, and hypothesises a
   likely cause.
3. **Records** the result for sign-off: a controller review PDF, an anomalies
   CSV with a disposition column, and a full audit trail.

The detection is deterministic and testable. The AI adds judgement on top but
never removes a flag, so nothing the detection surfaces is ever hidden from the
controller.

The data in this repository is illustrative sample data for a fictional entity.

---

## How an account is flagged

An account is flagged when **either** test is met.

**Materiality.** A large variance against prior month, budget, or forecast.
Flagged if the percentage change clears the account's band *and* the euro
change clears a floor, *or* the euro change is very large on its own.
Thresholds are set per account: tighter for Revenue and Personnel, looser for
discretionary costs. A near-zero benchmark is handled safely, no percentage is
computed against a tiny base, so a small account cannot produce a misleading
large percentage.

**Unusual for the account.** A value out of character with the account's own
12-month history, measured with a modified z-score (median and MAD, robust on
small samples and not distorted by the very outliers it is meant to detect). A
normally stable account that moves at all is flagged, because any change to a
fixed line is worth checking. This is what catches a small movement that a
threshold alone would miss.

---

## How the AI triages

For each flagged account the agent returns, as structured data:

- A **priority** (HIGH, MEDIUM, LOW) for the controller
- An **assessment**: timing (a one-off that reverses, such as an annual invoice
  or a conference) or permanent (a real step change)
- The **headline** number and the **benchmark story** (which comparison tells
  the real picture)
- A one-line **hypothesis** for the cause, to verify
- Its **confidence**

The agent returns both a readable narrative and a structured JSON block. The
JSON drives the PDF blocks and the CSV; the narrative gives the controller
something to read. Crucially, the agent must return every flagged account and
can never drop one, even if it thinks an item is noise, it stays on the list
marked low with its reasoning.

---

## Human review and sign-off

Detection and triage do not close the loop. The controller does.

When anomalies are flagged, the audit log sets `requires_review` to true. The
controller reads the PDF, then fills in the `disposition` column of the
anomalies CSV (accept, dismiss, or investigate) for each account, and records
the sign-off with:

```bash
python review.py "Your Name"
```

The script checks that every anomaly has a disposition and warns if any are
blank before recording who reviewed and when. The agent informs the decision;
the controller makes it, and the audit trail proves it was made.

---

## How to run

```bash
git clone https://github.com/cfpai0810/anomaly-detection-alert-agent.git
cd anomaly-detection-alert-agent
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
```

Add your Anthropic API key to a `.env` file:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Run:

```bash
python main.py
```

Three files are written to `output/`: the triage text, the controller review
PDF, and the anomalies CSV.

---

## Project structure

```
main.py                          Orchestrator
config.py                        Layer 1: thresholds, benchmarks, paths
review.py                        Controller sign-off with disposition check
requirements.txt

src/
  step1_data_loader.py           Layer 2: load and validate close and history
  step2_detection_engine.py      Layer 3: variance, materiality, volatility
  step3_triage_agent.py          Layer 4: prompt, Claude call, JSON parsing
  step4_output_writer.py         Layer 5: text, PDF, anomalies CSV, audit

data/
  close_actuals.csv              One close: account, actual, budget, prior, forecast
  account_history.csv            12 trailing months per account

docs/
  sample_review.pdf              Example controller review pack
  sample_anomalies.csv           Example disposition sheet
  sample_output.txt              Example triage narrative

output/                          Generated files (gitignored)
tests/
  test_pipeline.py               52 assertions across 9 test classes
```

---

## Architecture

**Core design rule:** the detection layer is deterministic. The same close data
always produces the same flags, and every flag is traceable to a specific rule.
The AI triage adds interpretation on top, treated as hypotheses for a human to
verify, never as the final word.

```
Two CSV inputs (close + history)
      |
      v
step1_data_loader.py     Load and validate. History ordered per account.
      |
      v
step2_detection_engine.py  Variance vs three benchmarks, three-part materiality
                           rule, modified z-score volatility context. Flag list.
      |
      v
step3_triage_agent.py    Build prompt with each flag and its context. Claude
                         returns a narrative plus structured JSON triage.
      |
      v
step4_output_writer.py   Triage text, controller review PDF, anomalies CSV with
                         a disposition column, and one audit record per run.
```

The two-layer split is the heart of the design. Deterministic detection
produces verifiable findings; the generative layer adds interpretability. This
separation of the maths from the judgement is a common pattern in production
anomaly-detection systems.

---

## Audit trail

Every run appends one record to `output/audit_log.jsonl`:

```json
{
  "run_id":           "2026-07-11T16:00:00+00:00",
  "project":          "anomaly-detection-alert-agent",
  "entity":           "Valencia Operations",
  "close_period":     "2026-06",
  "accounts_flagged": 3,
  "flagged_accounts": ["Marketing Spend", "IT Infrastructure", "Travel & Ent"],
  "close_hash":       "sha256:...",
  "history_hash":     "sha256:...",
  "output_file":      "output/close_triage_2026-07-11.txt",
  "pdf_file":         "output/close_review_2026-07-11.pdf",
  "csv_file":         "output/anomalies_2026-07-11.csv",
  "model":            "claude-sonnet-4-6",
  "input_tokens":     1600,
  "output_tokens":    900,
  "stop_reason":      "end_turn",
  "detection_flags":  [],
  "human_reviewed":   false,
  "requires_review":  true
}
```

Two separate hashes, one for the close and one for the history, mean it is
always possible to tell which input changed between runs. After sign-off, the
record also carries `reviewed_by`, `reviewed_at`, and the disposition counts.

---

## Test suite

52 assertions across 9 test classes. No real API calls. Runs in under a minute:

```bash
pytest tests/test_pipeline.py -v
```

The classes cover data loading, the near-zero variance guard, the three-part
materiality rule, per-account thresholds, the modified z-score (including its
robustness to outliers), the full detection integration, JSON parsing, account
name normalisation, and output writing with the audit trail.

---

## Tech stack

Python 3.11 · pandas · Anthropic Claude API · python-dotenv · reportlab ·
hashlib · pytest

---

## Related projects

| # | Project | Status |
|---|---------|--------|
| 1 | AI Variance Commentary Engine | Complete |
| 2 | Driver-Based Rolling Forecast Pipeline | Complete |
| 3 | Anomaly Detection and Alert Agent | This project |
| 4 | NL Scenario Modelling Copilot | Complete |
| 5 | Budget Challenge Assistant | Planned |
| 6 | Agentic Board Pack Generator | Planned |
| 7 | Planning-to-Warehouse-to-LLM Pipeline | Planned |
| 8 | 13-Week Cash Flow Forecasting Agent | Planned |
| 9 | Multi-Entity Consolidation and FX Engine | Planned |
| 10 | AI Governance Playbook | Planned |
