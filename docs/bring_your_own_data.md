### Your own data

The public web app runs on synthetic sample data only. To scan your
own month-end close, download this project and run it locally with
your own Anthropic API key.

One useful property: the scan itself needs no API key. Loading your
data, detecting anomalies, and adjusting the sensitivity all run
locally with no key. Only the AI triage step (the written assessment
of each flagged account) calls the API and needs a key.

---

## The two files

Both live in the `data/` folder.

- `close_actuals.csv` -- one month-end close for one entity, one row
  per account.
- `account_history.csv` -- the trailing monthly history per account,
  used for the statistical test.

---

## close_actuals.csv (the month being reviewed)

One row per account. Columns:

- `account` -- the account name (text, for example Revenue,
  Marketing Spend).
- `actual` -- the actual amount for the month (a number).
- `budget` -- the budgeted amount.
- `prior` -- last month's actual.
- `forecast` -- the latest forecast.

```text
account,actual,budget,prior,forecast
Revenue,1290000,1300000,1320000,1295000
Marketing Spend,143000,98000,96000,99000
IT Infrastructure,51000,45000,45000,45000
```

Amounts are plain numbers, entered without currency symbols; the
figures you enter are treated as being in the currency the tool
displays (the sample uses euros). The three benchmarks (budget,
prior, forecast) are what each account is measured against.

---

## account_history.csv (the trailing history)

The recent monthly values per account, so the tool can judge whether
the current month is unusual for that account. Columns:

- `account` -- the account name, matching the names in
  `close_actuals.csv`.
- `period` -- the month, as `YYYY-MM` (for example 2025-07).
- `value` -- the account's value that month (a number).

```text
account,period,value
Marketing Spend,2025-06,61000
Marketing Spend,2025-07,64000
Marketing Spend,2025-08,68000
```

The sample uses 12 trailing months per account. More history gives
the statistical test a more stable picture; too few months makes it
unreliable. Provide a consistent run of months for each account you
want the statistical test to cover.

---

## How the two files relate

The `account` names in `account_history.csv` should match the names
in `close_actuals.csv`. An account in the close with no matching
history is still scanned for materiality (the variance tests), but
it cannot get the statistical "unusual for this account" test, and
the tool notes the missing history.

You choose your own chart of accounts. The tool does not require a
fixed set of account names. Any accounts work.

---

## Sensitivity and custom thresholds (optional)

The tool flags an account two ways: a material variance against a
benchmark, and a value that is unusual for that account's own
history. Both are controlled by thresholds you can adjust in the
app (the sensitivity sliders for quick presets, or custom thresholds
for full control over the exact numbers).

- By default, every account uses one global set of thresholds.
- Optionally, you can give specific accounts their own thresholds
  (for example a tighter band for Revenue than for a discretionary
  cost). In the code these are the per-account overrides in
  `config.py`; an account without an override uses the global default.
  This is optional; the tool works with the global thresholds alone.

---

## Running it

You need your own Anthropic API key for the AI triage (the scan
itself does not need one). The app reads the key from the sidebar
for the session, and it can also be provided through the
`ANTHROPIC_API_KEY` environment variable.

From the project root:

```bash
pip install -r requirements.txt
streamlit run streamlit_app/Home.py
```

Open the close scan page, adjust the sensitivity if you want, and
read the flagged accounts. To get the AI triage (the written
assessment and the review pack), enter your key and run the triage.

You can also run the command-line version:

```bash
python main.py
```

which scans the data files and, with a key set, writes the triage
narrative, the anomalies CSV, and the review PDF to the `output/`
folder.

---

## Constraints worth knowing

- **The period format in the history is `YYYY-MM`** (for example
  2026-06), zero-padded, no day. The tool sorts periods as text, so
  a missing zero would sort incorrectly.
- **Amounts are plain numbers**, entered without currency symbols.
  The figures are treated as being in the currency the tool displays
  (the sample uses euros).
- **History depth matters for the statistical test.** The sample
  uses 12 months. With very little history, the "unusual for this
  account" test has too little to work with.
- **The account names must match between the two files** for an
  account to get the statistical test.
- **The scan is keyless; only the AI triage needs a key.** You can
  load, scan, and tune sensitivity with no key at all.

---

## What stays private, and what does not

Running locally, your data files stay on your machine, and the
public web app never sees them. The scan and the sensitivity tuning
happen entirely locally with no API call.

When you run the AI triage, what is sent to the Anthropic API is
the flagged accounts and their figures (the variances, the
benchmarks, the volatility signal) so the model can write its
assessment. Your full data file is not sent; only the flagged
accounts and the numbers the model needs to triage them. The call
goes to the Anthropic API under your own account and key, and
nothing goes to this project's authors or to the public app.
