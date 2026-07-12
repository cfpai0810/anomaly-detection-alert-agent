# =============================================================================
# step4_output_writer.py — Layer 5: Output Writing and Audit Trail
# =============================================================================
# Responsibilities:
#   - write_output():          write the triage narrative + audit record
#   - write_pdf():             controller review pack (added Step 5)
#   - export_anomalies_csv():  the disposition artefact (added Step 6)
#
# Knows about: file paths, audit logs, timestamps, PDF, CSV
# Does NOT know about: Claude, detection maths
# =============================================================================

import json
import hashlib
import re
from pathlib import Path
from datetime import datetime, timezone

from reportlab.lib.pagesizes import A4
from reportlab.lib.units     import cm
from reportlab.lib           import colors
from reportlab.lib.styles    import ParagraphStyle
from reportlab.lib.enums     import TA_CENTER, TA_RIGHT
from reportlab.platypus      import (
    SimpleDocTemplate, Paragraph, Spacer,
    HRFlowable, Table, TableStyle, KeepTogether
)

from config import (
    OUTPUT_DIR, AUDIT_LOG, CLOSE_FILE, HISTORY_FILE,
    DEFAULT_ENTITY, CLOSE_PERIOD, MODEL, BENCHMARKS,
)

# ── Page geometry and palette (same system as Projects 1 and 2) ───────────────
PAGE_W = A4[0] - 4 * cm

DARK_BLUE  = colors.HexColor("#1A3A5C")
MID_BLUE   = colors.HexColor("#2D6A9F")
LIGHT_BLUE = colors.HexColor("#EAF2FB")
FLAG_RED   = colors.HexColor("#A32D2D")
FLAG_BG    = colors.HexColor("#FFF0F0")
AMBER      = colors.HexColor("#854F0B")
AMBER_BG   = colors.HexColor("#FAEEDA")
GREEN      = colors.HexColor("#1D6B0F")
GREEN_BG   = colors.HexColor("#EAF3DE")
BODY_DARK  = colors.HexColor("#1A1A19")
MUTED      = colors.HexColor("#898781")
RULE_COLOR = colors.HexColor("#D3D1C7")
ROW_ALT    = colors.HexColor("#F8F7F2")
TBL_HEADER = colors.HexColor("#E6F1FB")

S_BODY    = ParagraphStyle("Body",   fontName="Helvetica", fontSize=10,
                textColor=BODY_DARK, leading=15)
S_META    = ParagraphStyle("Meta",   fontName="Helvetica", fontSize=8,
                textColor=MUTED, leading=12, alignment=TA_CENTER)
S_CARD    = ParagraphStyle("Card",   fontName="Helvetica", fontSize=9,
                textColor=BODY_DARK, leading=13)
S_NARR    = ParagraphStyle("Narr",   fontName="Helvetica", fontSize=9,
                textColor=BODY_DARK, leading=13)
S_ACCT    = ParagraphStyle("Acct",   fontName="Helvetica-Bold", fontSize=10,
                textColor=DARK_BLUE, leading=13)
S_HEAD    = ParagraphStyle("Head",   fontName="Helvetica-Bold", fontSize=9,
                textColor=MID_BLUE, leading=12)
S_TBL     = ParagraphStyle("Tbl",    fontName="Helvetica", fontSize=8,
                textColor=BODY_DARK, leading=10)
S_TBL_HDR = ParagraphStyle("TblHdr", fontName="Helvetica-Bold", fontSize=8,
                textColor=DARK_BLUE, leading=10)
S_TBL_NUM = ParagraphStyle("TblNum", fontName="Helvetica", fontSize=8,
                textColor=BODY_DARK, leading=10, alignment=TA_RIGHT)

PRIORITY_COLORS = {
    "HIGH":   (FLAG_RED, FLAG_BG),
    "MEDIUM": (AMBER,    AMBER_BG),
    "LOW":    (GREEN,    GREEN_BG),
}


def _normalise_account(name):
    """Normalise an account name for tolerant matching: lowercase, collapse
    whitespace, and treat '&' and 'and' as equivalent. Handles the case
    where the model rewords 'Travel & Ent' as 'Travel and Ent'."""
    import re as _re
    n = str(name).lower().strip()
    n = n.replace("&", " and ")
    n = _re.sub(r"\s+", " ", n)
    return n


def _build_triage_lookup(triage_json):
    """Build an account -> triage dict keyed by normalised name, so lookups
    tolerate the model rewording account names."""
    lookup = {}
    if triage_json:
        for t in triage_json:
            key = _normalise_account(t.get("account", ""))
            lookup[key] = t
    return lookup


def clean_markdown(text):
    """Strip markdown artefacts before PDF rendering. Escape & first."""
    text = text.replace("&", "&amp;")
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'^#{1,3}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^---+\s*$',   '', text, flags=re.MULTILINE)
    return text.strip()


def write_output(triage, flagged, flags, tok_in, tok_out, stop_reason):
    """
    Write the triage narrative to a timestamped text file and append one
    JSONL audit record. Two SHA256 hashes (close + history) mean it is
    always possible to tell which input changed between runs.
    """
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
        "run_id":           ts_log,
        "project":          "anomaly-detection-alert-agent",
        "entity":           DEFAULT_ENTITY,
        "close_period":     CLOSE_PERIOD,
        "accounts_flagged": len(flagged),
        "flagged_accounts": [f["account"] for f in flagged],
        "close_hash":       close_hash,
        "history_hash":     history_hash,
        "output_file":      str(out_path),
        "pdf_file":         None,
        "csv_file":         None,
        "model":            MODEL,
        "input_tokens":     tok_in,
        "output_tokens":    tok_out,
        "stop_reason":      stop_reason,
        "detection_flags":  flags,
        "human_reviewed":   False,
        "requires_review":  requires_review,
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


def _cover_block(entity, period, n_flagged, ts, tok_in, tok_out):
    rows = [
        [Paragraph(
            '<font color="white"><b>MONTH-END CLOSE ANOMALY REVIEW</b></font>',
            ParagraphStyle("CT", fontName="Helvetica-Bold", fontSize=15,
                textColor=colors.white, alignment=TA_CENTER))],
        [Paragraph(
            '<font color="#AACCEE">{}  ·  Close {}  ·  {} accounts flagged</font>'.format(
                entity, period, n_flagged),
            ParagraphStyle("CS", fontName="Helvetica", fontSize=9,
                textColor=colors.HexColor("#AACCEE"), alignment=TA_CENTER))],
        [Paragraph(
            '<font color="#6699BB">Generated {}  ·  {:,}/{:,} tokens  ·  AI triage</font>'.format(
                ts[:10], tok_in, tok_out),
            ParagraphStyle("CM", fontName="Helvetica", fontSize=8,
                textColor=colors.HexColor("#6699BB"), alignment=TA_CENTER))],
    ]
    t = Table(rows, colWidths=[PAGE_W])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), DARK_BLUE),
        ("TOPPADDING",    (0, 0), (0, 0),   16),
        ("BOTTOMPADDING", (0, 0), (0, 0),   5),
        ("TOPPADDING",    (0, 1), (0, 1),   3),
        ("BOTTOMPADDING", (0, 1), (0, 1),   3),
        ("TOPPADDING",    (0, 2), (0, 2),   3),
        ("BOTTOMPADDING", (0, 2), (0, 2),   14),
        ("LEFTPADDING",   (0, 0), (-1, -1), 20),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 20),
    ]))
    return t


def _section_header(title):
    t = Table([[Paragraph(
        '<font color="white"><b>{}</b></font>'.format(title),
        ParagraphStyle("SH", fontName="Helvetica-Bold", fontSize=11,
            textColor=colors.white, leading=14))]], colWidths=[PAGE_W])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), MID_BLUE),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def _anomaly_card(flag, triage_lookup):
    """One card per flagged account: header with priority badge, variance
    table, and the agent's assessment. Returns a KeepTogether flowable so a
    card never splits across a page break."""
    account     = flag["account"]
    actual      = flag["actual"]
    comparisons = flag["comparisons"]
    volatility  = flag["volatility"]

    t = triage_lookup.get(_normalise_account(account), {})
    priority   = t.get("priority", "").upper() or "REVIEW"
    assessment = t.get("assessment", "")
    hypothesis = t.get("hypothesis", "")
    confidence = t.get("confidence", "")

    tc, bg = PRIORITY_COLORS.get(priority, (MUTED, ROW_ALT))

    hdr = Table([[
        Paragraph(
            '<b>{}</b>  <font size="8" color="#898781">actual EUR {:,.0f}</font>'.format(
                account, actual),
            ParagraphStyle("h", fontName="Helvetica", fontSize=11,
                textColor=DARK_BLUE, leading=14)),
        Paragraph('<b>{}</b>'.format(priority),
            ParagraphStyle("p", fontName="Helvetica-Bold", fontSize=9,
                textColor=tc, alignment=TA_RIGHT, leading=12)),
    ]], colWidths=[PAGE_W - 3*cm, 3*cm])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))

    rows = [[
        Paragraph("<b>vs</b>", S_TBL_HDR),
        Paragraph("<b>Benchmark</b>", S_TBL_HDR),
        Paragraph("<b>Change</b>", S_TBL_HDR),
        Paragraph("<b>%</b>", S_TBL_HDR),
        Paragraph("<b>Material</b>", S_TBL_HDR),
    ]]
    for bench in BENCHMARKS:
        c = comparisons[bench]
        pct = "{:+.1%}".format(c["pct"]) if c["pct"] is not None else "n/a"
        mat = "yes" if c["material"] else "-"
        rows.append([
            Paragraph(bench, S_TBL),
            Paragraph("{:,.0f}".format(c["benchmark_value"]), S_TBL_NUM),
            Paragraph("{:+,.0f}".format(c["dollar"]), S_TBL_NUM),
            Paragraph(pct, S_TBL_NUM),
            Paragraph(mat, S_TBL),
        ])
    vt = Table(rows, colWidths=[2*cm, 3*cm, 3*cm, 2.5*cm, PAGE_W - 10.5*cm])
    vt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TBL_HEADER),
        ("LINEBELOW",  (0, 0), (-1, 0), 0.75, MID_BLUE),
        ("LINEBELOW",  (0, 1), (-1, -1), 0.25, RULE_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))

    if assessment or hypothesis:
        note_text = ""
        if assessment:
            note_text += '<font color="#2D6A9F"><b>{}</b></font>. '.format(
                assessment.capitalize())
        if hypothesis:
            note_text += '<b>Likely cause:</b> ' + clean_markdown(hypothesis) + " "
        extras = []
        if confidence:
            extras.append("confidence {}".format(confidence))
        extras.append("volatility {}".format(volatility))
        note_text += '<font color="#898781">({})</font>'.format(", ".join(extras))
        note = Paragraph(note_text, S_CARD)
    else:
        note = Paragraph(
            '<font color="#898781">volatility {}</font>'.format(volatility), S_CARD)

    return KeepTogether([hdr, Spacer(1, 0.1*cm), vt, Spacer(1, 0.12*cm),
                         note, Spacer(1, 0.35*cm)])


def update_audit_pdf(pdf_path):
    """Record the PDF path in the most recent audit record."""
    if not AUDIT_LOG.exists():
        return
    lines = AUDIT_LOG.read_text(encoding="utf-8").strip().split("\n")
    if not lines or not lines[-1].strip():
        return
    last = json.loads(lines[-1])
    last["pdf_file"] = str(pdf_path)
    lines[-1] = json.dumps(last)
    AUDIT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _methodology_box():
    """Explain the detection criteria: what makes an account an anomaly."""
    intro = Paragraph(
        "An account is flagged for review when <b>either</b> test below is met. "
        "Detection is deterministic: the same close data always produces the same flags.",
        S_BODY)

    rows = [
        [Paragraph("<b>Test</b>", S_TBL_HDR),
         Paragraph("<b>What it catches</b>", S_TBL_HDR),
         Paragraph("<b>How it works</b>", S_TBL_HDR)],
        [Paragraph("<b>Materiality</b>", S_TBL),
         Paragraph("Large variances vs prior month, budget, or forecast", S_TBL),
         Paragraph("Flagged if the percentage clears the account band <b>and</b> the "
                   "euro change clears the floor, <b>or</b> the euro change is very "
                   "large on its own. Thresholds are per account: tighter for Revenue "
                   "and Personnel, looser for discretionary costs.", S_TBL)],
        [Paragraph("<b>Unusual for<br/>the account</b>", S_TBL),
         Paragraph("A value out of character with the account's own history", S_TBL),
         Paragraph("A modified z-score (median and MAD, robust on small samples) versus "
                   "the trailing 12 months. A normally stable account that moves at all "
                   "is flagged, because any change to a fixed line is worth checking.", S_TBL)],
    ]
    t = Table(rows, colWidths=[2.6*cm, 4.2*cm, PAGE_W - 6.8*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  TBL_HEADER),
        ("LINEBELOW",     (0, 0), (-1, 0),  0.75, MID_BLUE),
        ("LINEBELOW",     (0, 1), (-1, -1), 0.25, RULE_COLOR),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))

    note = Paragraph(
        "<font color='#898781'>Near-zero bases are handled safely: no percentage is "
        "computed against a tiny or zero benchmark, so a small account cannot produce a "
        "misleading large percentage. The AI triage that follows adds judgement "
        "(priority, timing versus permanent, a likely cause) but never removes a flag.</font>",
        S_CARD)

    return [intro, Spacer(1, 0.2*cm), t, Spacer(1, 0.15*cm), note]


def _narrative_highlights(triage_narrative):
    """Pull the opening summary sentence and the RECOMMENDED ACTIONS line
    from the triage narrative. The per-account detail already lives in the
    cards, so the notes section shows only these two highlights."""
    text = clean_markdown(triage_narrative)
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    summary = ""
    for line in lines:
        # first real sentence that is not a section marker
        if line.upper() not in ("ANOMALY TRIAGE",) and not line.endswith("]"):
            summary = line
            break

    rec = ""
    m = re.search(r'RECOMMENDED ACTIONS:?\s*(.+)', text, re.DOTALL | re.IGNORECASE)
    if m:
        rec = m.group(1).strip().split("\n")[0]

    return summary, rec


def _triage_narrative_section(triage_json, summary):
    """
    Build the elaborate AI triage section from the structured triage. One
    three-line block per account:
      1. account name, headline number, priority badge
      2. assessment (colour-coded: red permanent, amber timing) + the story
         (which benchmark tells the real story)
      3. likely cause + confidence
    The bottom rule is colour-coded by priority for quick visual scanning.
    Built from the JSON, not from parsing prose, so it is reliable.
    """
    elements = []

    if summary:
        elements.append(Paragraph("<b>Summary.</b> " + clean_markdown(summary), S_BODY))
        elements.append(Spacer(1, 0.3 * cm))

    prio_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    ordered = sorted(triage_json,
                     key=lambda x: prio_order.get(str(x.get("priority", "")).upper(), 3))

    for t in ordered:
        account    = t.get("account", "")
        priority   = str(t.get("priority", "REVIEW")).upper()
        assessment = str(t.get("assessment", "")).strip()
        headline   = clean_markdown(str(t.get("headline", "")).strip())
        story      = clean_markdown(str(t.get("story", "")).strip())
        hypothesis = clean_markdown(str(t.get("hypothesis", "")).strip())
        confidence = str(t.get("confidence", "")).strip()

        tc, bg = PRIORITY_COLORS.get(priority, (MUTED, ROW_ALT))
        assess_color = "#A32D2D" if assessment.lower() == "permanent" else "#854F0B"
        row_bg = bg if priority == "HIGH" else LIGHT_BLUE

        left  = Paragraph(account, S_ACCT)
        head  = Paragraph(headline, S_HEAD)
        badge = Table([[Paragraph(
            '<font color="white"><b>{}</b></font>'.format(priority),
            ParagraphStyle("bdg", fontName="Helvetica-Bold", fontSize=8,
                textColor=colors.white, alignment=TA_CENTER, leading=10))]],
            colWidths=[1.9*cm])
        badge.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), tc),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))

        line2_bits = []
        if assessment:
            line2_bits.append('<font color="{}"><b>{}.</b></font>'.format(
                assess_color, assessment.capitalize()))
        if story:
            line2_bits.append(story + ".")
        line2 = Paragraph(" ".join(line2_bits), S_NARR)

        line3_bits = []
        if hypothesis:
            line3_bits.append("<b>Likely cause:</b> " + hypothesis + ".")
        if confidence:
            line3_bits.append('<font color="#898781">Confidence {}.</font>'.format(confidence))
        line3 = Paragraph(" ".join(line3_bits), S_NARR)

        row = Table(
            [[left, head, badge], [line2, "", ""], [line3, "", ""]],
            colWidths=[PAGE_W - 5.9*cm, 4*cm, 1.9*cm])
        row.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), row_bg),
            ("SPAN",          (0, 1), (2, 1)),
            ("SPAN",          (0, 2), (2, 2)),
            ("VALIGN",        (0, 0), (-1, 0), "MIDDLE"),
            ("ALIGN",         (1, 0), (1, 0), "RIGHT"),
            ("TOPPADDING",    (0, 0), (-1, 0), 5),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 3),
            ("TOPPADDING",    (0, 1), (-1, 1), 0),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 3),
            ("TOPPADDING",    (0, 2), (-1, 2), 0),
            ("BOTTOMPADDING", (0, 2), (-1, 2), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("LINEBELOW",     (0, -1), (-1, -1), 1, tc),
        ]))
        elements.append(KeepTogether([row, Spacer(1, 0.18 * cm)]))

    return elements


def write_pdf(triage_narrative, flagged, triage_json, tok_in, tok_out):
    """
    Build the controller review pack:
      1. Cover (entity, period, count, metadata)
      2. Anomaly cards, one per flagged account, sorted by priority
      3. The triage narrative
      4. Footer
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    now      = datetime.now(timezone.utc)
    ts_file  = now.strftime("%Y-%m-%d_%H-%M-%S")
    ts_log   = now.isoformat()
    pdf_path = OUTPUT_DIR / "close_review_{}.pdf".format(ts_file)

    triage_lookup = _build_triage_lookup(triage_json)

    # Sort cards by priority HIGH, MEDIUM, LOW, then anything else
    prio_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    def card_key(f):
        p = triage_lookup.get(_normalise_account(f["account"]), {}).get("priority", "").upper()
        return prio_order.get(p, 3)
    ordered = sorted(flagged, key=card_key)

    story = []
    story.append(_cover_block(DEFAULT_ENTITY, CLOSE_PERIOD, len(flagged),
                              ts_log, tok_in, tok_out))
    story.append(Spacer(1, 0.4 * cm))

    # AI triage assessment — the judgement, concisely highlighted
    summary, _ = _narrative_highlights(triage_narrative)
    story.append(_section_header("AI TRIAGE ASSESSMENT"))
    story.append(Spacer(1, 0.25 * cm))
    if triage_json:
        for element in _triage_narrative_section(triage_json, summary):
            story.append(element)
    elif summary:
        story.append(Paragraph("<b>Summary.</b> " + summary, S_BODY))
    story.append(Spacer(1, 0.25 * cm))

    # The supporting numbers, one card per flagged account
    story.append(_section_header("VARIANCE DETAIL BY ACCOUNT"))
    story.append(Spacer(1, 0.25 * cm))
    for f in ordered:
        story.append(_anomaly_card(f, triage_lookup))

    # How the judgement is made
    story.append(_section_header("HOW ACCOUNTS ARE FLAGGED"))
    story.append(Spacer(1, 0.2 * cm))
    for element in _methodology_box():
        story.append(element)
    story.append(Spacer(1, 0.4 * cm))

    story.append(HRFlowable(width="100%", thickness=0.5, color=RULE_COLOR))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(
        "AI Anomaly Detection and Alert Agent  ·  {}  ·  {}  ·  "
        "Detection is deterministic; triage is an AI hypothesis for human review.".format(
            MODEL, ts_log[:10]),
        S_META))

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
        title="Close Anomaly Review - {}".format(CLOSE_PERIOD),
        author="AI Anomaly Detection and Alert Agent",
    )
    doc.build(story)
    update_audit_pdf(pdf_path)

    print("[OK] PDF written")
    print("     PDF:  {}".format(pdf_path.name))
    print("     Size: {:.1f} KB".format(pdf_path.stat().st_size / 1024))
    return pdf_path


def update_audit_csv(csv_path):
    """Record the CSV path in the most recent audit record."""
    if not AUDIT_LOG.exists():
        return
    lines = AUDIT_LOG.read_text(encoding="utf-8").strip().split("\n")
    if not lines or not lines[-1].strip():
        return
    last = json.loads(lines[-1])
    last["csv_file"] = str(csv_path)
    lines[-1] = json.dumps(last)
    AUDIT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_anomalies_csv(flagged, triage_json):
    """
    Export the anomaly list with a disposition column: the governance
    artefact the controller works from. Each row merges the deterministic
    detection data (variances, volatility) with the agent's triage
    (priority, assessment, hypothesis) and leaves disposition and
    reviewed_by blank for the controller to complete.

    Uses pandas to_csv so any commas inside the agent's text are quoted
    correctly rather than breaking the columns.
    """
    import pandas as pd

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    now      = datetime.now(timezone.utc)
    ts_file  = now.strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = OUTPUT_DIR / "anomalies_{}.csv".format(ts_file)

    triage_lookup = _build_triage_lookup(triage_json)

    rows = []
    for f in flagged:
        account = f["account"]
        t       = triage_lookup.get(_normalise_account(account), {})
        c       = f["comparisons"]

        def pct_str(bench):
            p = c[bench]["pct"]
            return "" if p is None else "{:.1%}".format(p)

        rows.append({
            "account":             account,
            "actual":              f["actual"],
            "vs_prior_eur":        c["prior"]["dollar"],
            "vs_prior_pct":        pct_str("prior"),
            "vs_budget_eur":       c["budget"]["dollar"],
            "vs_budget_pct":       pct_str("budget"),
            "vs_forecast_eur":     c["forecast"]["dollar"],
            "vs_forecast_pct":     pct_str("forecast"),
            "material_benchmarks": "+".join(f["material_benchmarks"]) if f["material_benchmarks"] else "",
            "volatility":          f["volatility"],
            "agent_priority":      t.get("priority", ""),
            "agent_assessment":    t.get("assessment", ""),
            "agent_headline":      t.get("headline", ""),
            "agent_story":         t.get("story", ""),
            "agent_hypothesis":    t.get("hypothesis", ""),
            "agent_confidence":    t.get("confidence", ""),
            "disposition":         "",
            "reviewed_by":         "",
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    update_audit_csv(csv_path)

    print("[OK] Anomalies CSV exported")
    print("     CSV:  {}".format(csv_path.name))
    print("     Rows: {} (disposition column blank for controller)".format(len(df)))
    return csv_path
