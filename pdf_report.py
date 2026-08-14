"""
Per-case PDF "Evidence Report" generator.

Produces a single downloadable PDF for one analysis: the message (with the
same inline tactic highlighting used in the web UI), the verdict, the full
statistics, and — the point of this module — a plain-English narrative
explanation of *why* the model reached that verdict, not just the numbers.
"""

import io
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

INK = colors.HexColor("#232323")
NAVY = colors.HexColor("#2b3a55")
PAPER = colors.HexColor("#f2ecd8")
PAPER_DARK = colors.HexColor("#e4dcc3")
MUTED = colors.HexColor("#5c5747")
FOOTNOTE = colors.HexColor("#8a836b")

VERDICT_COLORS = {
    "SPAM": colors.HexColor("#a13328"),
    "SUSPICIOUS": colors.HexColor("#8a5d10"),
    "HAM": colors.HexColor("#2f6844"),
}
VERDICT_LABELS = {"SPAM": "SPAM", "SUSPICIOUS": "SUSPICIOUS", "HAM": "CLEARED"}

TACTIC_COLORS = {
    "urgency": colors.HexColor("#c98a1f"),
    "scarcity": colors.HexColor("#8b5fa3"),
    "money_bait": colors.HexColor("#4f9268"),
    "false_authority": colors.HexColor("#3f6fa3"),
    "pressure_to_act": colors.HexColor("#b34c40"),
    "excess_caps": colors.HexColor("#a68b2c"),
    "excess_punct": colors.HexColor("#7a6a9c"),
    "has_url": colors.HexColor("#b34c40"),
    "has_phone": colors.HexColor("#3f6fa3"),
}
TACTIC_LABELS = {
    "urgency": "Urgency", "scarcity": "Scarcity", "money_bait": "Money bait",
    "false_authority": "False authority", "pressure_to_act": "Pressure to act",
    "excess_caps": "Excess CAPS", "excess_punct": "Excess punctuation",
    "has_url": "Contains a link", "has_phone": "Contains a phone #",
}

styles = getSampleStyleSheet()

eyebrow_style = ParagraphStyle(
    "Eyebrow", fontSize=9.5, leading=12, textColor=NAVY,
    fontName="Helvetica-Bold", spaceAfter=2,
)
title_style = ParagraphStyle(
    "ReportTitle", fontSize=22, leading=26, textColor=INK,
    fontName="Helvetica-Bold", spaceAfter=4,
)
meta_style = ParagraphStyle(
    "Meta", fontSize=9.5, leading=13, textColor=MUTED, spaceAfter=10,
)
section_style = ParagraphStyle(
    "Section", fontSize=11.5, leading=14, textColor=NAVY,
    fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=8,
)
body_style = ParagraphStyle(
    "Body", fontSize=10.5, leading=16, textColor=INK, spaceAfter=9,
    alignment=TA_LEFT,
)
mono_style = ParagraphStyle(
    "Mono", fontName="Courier", fontSize=10, leading=16, textColor=INK,
)
footnote_style = ParagraphStyle(
    "Footnote", fontSize=8, leading=11, textColor=FOOTNOTE,
)
tactic_row_style = ParagraphStyle(
    "TacticRow", fontSize=9.5, leading=13, textColor=INK,
)


def _highlighted_message_xml(text, matches):
    """Build reportlab mini-markup with colored highlights matching the web UI."""
    if not matches:
        return xml_escape(text)
    spans = sorted(matches, key=lambda m: m["start"])
    out = []
    cursor = 0
    for m in spans:
        if m["start"] < cursor:
            continue  # skip overlaps, mirrors frontend behavior
        out.append(xml_escape(text[cursor:m["start"]]))
        color = TACTIC_COLORS.get(m["tactic"], colors.HexColor("#999999"))
        chunk = xml_escape(text[m["start"]:m["end"]])
        hexcode = "#" + color.hexval()[2:]
        out.append(f'<font backColor="{hexcode}" color="#232323">{chunk}</font>')
        cursor = m["end"]
    out.append(xml_escape(text[cursor:]))
    return "".join(out)


def build_case_report_pdf(entry: dict, username: str) -> bytes:
    """
    entry: dict with id, text, verdict, confidence, text_model_score,
           tactic_score, tactics (dict), matches (list), created_at, narrative (list[str])
    Returns raw PDF bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        title=f"Evidence Report — Case {entry['id']}",
        author="Message Forensics",
    )

    verdict = entry["verdict"]
    vcolor = VERDICT_COLORS.get(verdict, INK)
    vlabel = VERDICT_LABELS.get(verdict, verdict)

    story = []

    # ---- Header ----
    story.append(Paragraph("MESSAGE FORENSICS — EVIDENCE REPORT", eyebrow_style))
    story.append(Paragraph(f"Case No. {int(entry['id']):06d}", title_style))
    try:
        when = datetime.fromisoformat(entry["created_at"]).strftime("%B %d, %Y at %H:%M UTC")
    except Exception:
        when = entry["created_at"]
    story.append(Paragraph(
        f"Investigator: <b>{xml_escape(username)}</b> &nbsp;&nbsp;|&nbsp;&nbsp; Filed: {when}",
        meta_style
    ))
    story.append(HRFlowable(width="100%", thickness=0.6, color=PAPER_DARK, spaceAfter=8))

    # ---- Verdict badge ----
    badge = Table(
        [[Paragraph(f"VERDICT: {vlabel}", ParagraphStyle(
            "Badge", fontSize=13, leading=16, textColor=vcolor,
            fontName="Helvetica-Bold", alignment=TA_CENTER))]],
        colWidths=[2.6 * inch],
    )
    badge.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 1.4, vcolor),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(badge)
    story.append(Spacer(1, 12))

    # ---- Evidence (highlighted message) ----
    story.append(Paragraph("EVIDENCE — MESSAGE ANALYZED", section_style))
    evidence_xml = _highlighted_message_xml(entry["text"], entry.get("matches", []))
    evidence_table = Table(
        [[Paragraph(evidence_xml, mono_style)]],
        colWidths=[doc.width],
    )
    evidence_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PAPER),
        ("BOX", (0, 0), (-1, -1), 0.6, PAPER_DARK),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(evidence_table)

    # ---- Statistics ----
    story.append(Paragraph("STATISTICS", section_style))
    stat_rows = [
        ["Overall confidence", f"{round(entry['confidence']*100)}%",
         "Word-pattern score", f"{round(entry['text_model_score']*100)}%"],
        ["Tactic score", f"{round(entry['tactic_score']*100)}%",
         "Verdict", vlabel],
    ]
    stat_table = Table(stat_rows, colWidths=[doc.width*0.28, doc.width*0.22, doc.width*0.28, doc.width*0.22])
    stat_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica"),
        ("FONTNAME", (1, 0), (1, -1), "Courier-Bold"),
        ("FONTNAME", (3, 0), (3, -1), "Courier-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
        ("TEXTCOLOR", (2, 0), (2, -1), MUTED),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, PAPER_DARK),
    ]))
    story.append(stat_table)
    story.append(Spacer(1, 4))

    # ---- Tactic log ----
    tactics = entry.get("tactics", {})
    if tactics:
        story.append(Paragraph("TACTIC LOG", section_style))
        rows = sorted(tactics.items(), key=lambda kv: -kv[1])
        tlog_data = []
        for key, val in rows:
            label = TACTIC_LABELS.get(key, key)
            color = TACTIC_COLORS.get(key, colors.HexColor("#888888"))
            pct = round(val * 100)
            swatch = Table([[""]], colWidths=[9], rowHeights=[9])
            swatch.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), color)]))
            tlog_data.append([swatch, Paragraph(label, tactic_row_style), f"{pct}%"])
        tlog_table = Table(tlog_data, colWidths=[16, doc.width - 16 - 50, 50])
        tlog_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("ALIGN", (2, 0), (2, -1), "RIGHT"),
            ("FONTNAME", (2, 0), (2, -1), "Courier-Bold"),
            ("FONTSIZE", (2, 0), (2, -1), 9.5),
        ]))
        story.append(tlog_table)

    # ---- Narrative explanation (the point of this report) ----
    story.append(Paragraph("HOW WE REACHED THIS VERDICT", section_style))
    narrative = entry.get("narrative") or []
    for para in narrative:
        story.append(Paragraph(xml_escape(para), body_style))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=PAPER_DARK, spaceAfter=6))
    story.append(Paragraph(
        "Generated by Message Forensics (Hybrid Spam Detector). The tactic layer is rule-based on top of "
        "a trained ML model; this report explains the model's own reasoning and is not a guarantee. When "
        "in doubt about a real message, verify independently rather than acting on it directly.",
        footnote_style
    ))

    doc.build(story)
    return buf.getvalue()
