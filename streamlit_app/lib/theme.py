# =============================================================================
# lib/theme.py — one visual identity, shared with the PDF
# =============================================================================
# The exact palette the PDFs already use (step4_output_writer), so the web app
# and the report it produces read as a single product. Priority colours do real
# work here as badges: red HIGH, amber MEDIUM, green LOW.
#
# Typography: Source Serif 4 for headings (institutional authority) and Inter
# for body, UI, and data (screen-first, with tabular figures so numbers align
# in tables). Both are free Google Fonts, loaded by CSS import.
# =============================================================================

# Core palette — identical to the PDF (step4_output_writer.py)
DARK_BLUE  = "#1A3A5C"
MID_BLUE   = "#2D6A9F"
LIGHT_BLUE = "#EAF2FB"
GREEN      = "#1D6B0F"
AMBER      = "#854F0B"
AMBER_BG   = "#FAEEDA"
FLAG_RED   = "#A32D2D"
BODY_DARK  = "#1A1A19"
MUTED      = "#898781"
RULE       = "#D3D1C7"
NEAR_WHITE = "#FBFAF7"

# Status badge colours (background, text) — the governance state made visible
BADGE = {
    "draft":    (LIGHT_BLUE, DARK_BLUE),
    "approved": ("#E1F5EE",  GREEN),
    "locked":   ("#E6F1FB",  MID_BLUE),
    "refused":  ("#F7E4E4",  FLAG_RED),
    "caution":  (AMBER_BG,   AMBER),
}

# Priority badge colours — matches the PDF review pack
PRIORITY_BADGE = {
    "HIGH":   ("#FFF0F0", FLAG_RED),
    "MEDIUM": (AMBER_BG,  AMBER),
    "LOW":    ("#EAF3DE", GREEN),
}

# Font families (referenced in CSS and available to the SVG diagram)
SERIF = "'Source Serif 4', Georgia, 'Times New Roman', serif"
SANS  = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"


def inject_css():
    """Return the app's CSS, including the font imports. Kept in one place so
    every page is consistent."""
    return f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap');

      /* Body, UI, and data: Inter, with tabular figures so numbers align */
      html, body, [class*="css"], .stMarkdown, .stText, p, span, div, label,
      input, button, table, td, th {{
        font-family: {SANS};
        font-feature-settings: "tnum" 1, "cv05" 1;
      }}

      /* Headings: Source Serif 4 for institutional authority */
      h1, h2, h3, h4 {{
        font-family: {SERIF};
        color: {DARK_BLUE};
        letter-spacing: -0.01em;
        font-weight: 600;
      }}

      /* The app header band */
      .sc-header {{
        background: {DARK_BLUE};
        color: white;
        padding: 22px 26px;
        border-radius: 10px;
        margin-bottom: 8px;
      }}
      .sc-header h1 {{
        font-family: {SERIF};
        color: white; margin: 0; font-size: 1.7rem; font-weight: 700;
        letter-spacing: -0.02em;
      }}
      .sc-header p  {{
        font-family: {SANS};
        color: #AACCEE; margin: 6px 0 0; font-size: 0.92rem; font-weight: 400;
      }}

      /* Status badge */
      .sc-badge {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        font-family: {SANS};
      }}

      /* Honest key-disclosure note */
      .sc-keynote {{
        font-size: 0.8rem;
        color: {MUTED};
        border-left: 2px solid {RULE};
        padding-left: 10px;
        margin-top: 6px;
        font-family: {SANS};
      }}

      /* Triage card header: badge + account name on one line */
      .sc-card-header {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 2px;
      }}
      .sc-card-account {{
        font-family: {SERIF};
        font-weight: 600;
        font-size: 1.05rem;
        color: {DARK_BLUE};
      }}

      /* Triage narrative callout — matches the portfolio pattern */
      .sc-narrative {{
        background: {LIGHT_BLUE};
        border-left: 3px solid {MID_BLUE};
        border-radius: 4px;
        padding: 14px 18px;
        margin: 8px 0;
        font-size: 0.92rem;
        line-height: 1.6;
        color: {BODY_DARK};
        font-family: {SANS};
      }}

      /* Card field labels (Headline, Assessment, etc.) */
      .sc-card-field {{
        margin: 4px 0;
        font-size: 0.92rem;
        line-height: 1.5;
        color: {BODY_DARK};
        font-family: {SANS};
      }}
      .sc-card-label {{
        font-weight: 600;
        color: {DARK_BLUE};
      }}

      /* Placeholder region for a future phase */
      .sc-placeholder {{
        border: 1px dashed {RULE};
        border-radius: 8px;
        padding: 18px 20px;
        color: {MUTED};
        font-size: 0.88rem;
        text-align: center;
        margin: 8px 0;
      }}
    </style>
    """


def badge_html(status):
    """Return an HTML status badge for one of the governance states."""
    status = status.lower()
    bg, fg = BADGE.get(status, BADGE["draft"])
    label = status.upper()
    return (f'<span class="sc-badge" style="background:{bg};color:{fg};">'
            f'{label}</span>')


_VALID_PRIORITIES = frozenset({"HIGH", "MEDIUM", "LOW"})


def priority_badge_html(priority):
    """Return an HTML badge for a priority level (HIGH, MEDIUM, LOW).

    Only known priority values are rendered inside the badge HTML.
    An unexpected value is escaped to plain text so model output
    cannot inject markup through the priority field.
    """
    import html as _html
    label = str(priority).upper()
    if label not in _VALID_PRIORITIES:
        return _html.escape(label)
    bg, fg = PRIORITY_BADGE[label]
    return (f'<span class="sc-badge" style="background:{bg};color:{fg};">'
            f'{label}</span>')
