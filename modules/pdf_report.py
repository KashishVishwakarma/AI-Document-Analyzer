"""
modules/pdf_report.py
Professional White & Blue PDF Report Generator for AI Document Analyzer.
Uses ReportLab to generate an executive-style multi-page PDF summary.
"""

import html
from io import BytesIO
from typing import Dict, List, Any

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle
)
from reportlab.lib.enums import TA_LEFT


def clean_for_pdf(text: str) -> str:
    """Escapes XML entities to prevent ReportLab crashes."""
    if not text:
        return ""
    escaped = html.escape(str(text))
    return escaped.replace("\n", "<br/>")


def generate_pdf_report(
    filename: str,
    stats: Dict[str, Any],
    summary_data: Dict[str, Any],
    teacher_data: Dict[str, Any],
    quiz_data: List[Dict[str, Any]],
    flashcards_data: List[Dict[str, str]],
    nlp_data: Dict[str, Any],
    wordcloud_buf: BytesIO | None
) -> BytesIO:
    """
    Generates a structured, professional PDF document in White & Royal Blue styling.
    """
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    PRIMARY_BLUE = colors.HexColor("#1E40AF")
    LIGHT_BG = colors.HexColor("#EFF6FF")
    TEXT_DARK = colors.HexColor("#0F172A")
    TEXT_MUTED = colors.HexColor("#475569")
    BORDER_COLOR = colors.HexColor("#BFDBFE")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=PRIMARY_BLUE,
        alignment=TA_LEFT,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=TEXT_MUTED,
        alignment=TA_LEFT,
        spaceAfter=12
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=PRIMARY_BLUE,
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=TEXT_DARK,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        "DocBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=TEXT_DARK,
        leftIndent=14,
        spaceAfter=4
    )

    analogy_style = ParagraphStyle(
        "DocAnalogy",
        parent=styles["Italic"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1D4ED8"),
        leftIndent=10,
        spaceAfter=4
    )

    story = []

    # 1. HEADER BANNER TABLE
    header_content = [
        [
            Paragraph("<b>🧠 AI DOCUMENT ANALYZER</b>", title_style),
            Paragraph(f"<b>Document:</b> {clean_for_pdf(filename)}<br/><b>Words:</b> {stats.get('word_count', 0)} | <b>Est. Read Time:</b> {stats.get('reading_time_min', 1)} min", subtitle_style)
        ]
    ]
    header_table = Table(header_content, colWidths=[3.2 * inch, 4.0 * inch])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 2, PRIMARY_BLUE),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 10))

    # 2. EXECUTIVE SUMMARY & KEY TAKEAWAYS
    story.append(Paragraph("📑 Executive Summary", h2_style))
    exec_text = summary_data.get("executive_summary", "Summary unavailable.")
    story.append(Paragraph(clean_for_pdf(exec_text), body_style))
    story.append(Spacer(1, 6))

    takeaways = summary_data.get("key_takeaways", [])
    if takeaways:
        story.append(Paragraph("<b>Key Takeaways:</b>", body_style))
        for t in takeaways:
            story.append(Paragraph(f"• {clean_for_pdf(t)}", bullet_style))
    story.append(Spacer(1, 10))

    # 3. TEACHER-STYLE EXPLANATION
    story.append(Paragraph("👨‍🏫 Teacher-Style Explanation", h2_style))
    big_pic = teacher_data.get("big_picture", "")
    if big_pic:
        story.append(Paragraph(f"<b>💡 The Big Picture:</b> {clean_for_pdf(big_pic)}", body_style))
        story.append(Spacer(1, 6))

    concepts = teacher_data.get("concepts", [])
    if concepts:
        story.append(Paragraph("<b>🔍 Core Concepts Demystified:</b>", body_style))
        for c in concepts:
            c_name = clean_for_pdf(c.get("concept", ""))
            c_plain = clean_for_pdf(c.get("plain_english", ""))
            c_analogy = clean_for_pdf(c.get("analogy", ""))
            story.append(Paragraph(f"<b>• {c_name}:</b> {c_plain}", bullet_style))
            story.append(Paragraph(f"<i>Analogy: {c_analogy}</i>", analogy_style))
            story.append(Spacer(1, 3))
    story.append(Spacer(1, 10))

    # 4. QUIZ QUESTIONS (MCQ) & ANSWERS
    if quiz_data:
        story.append(Paragraph("🎯 Quiz Questions (MCQ)", h2_style))
        for q in quiz_data:
            q_num = q.get("id", "")
            q_text = clean_for_pdf(q.get("question", ""))
            story.append(Paragraph(f"<b>Q{q_num}. {q_text}</b>", body_style))
            for opt in q.get("options", []):
                story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{clean_for_pdf(opt)}", bullet_style))
            correct = clean_for_pdf(q.get("correct", ""))
            ans_text = clean_for_pdf(q.get("correct_answer", ""))
            explanation = clean_for_pdf(q.get("explanation", ""))
            story.append(Paragraph(f"<b>Correct Answer:</b> Option {correct} ({ans_text})", bullet_style))
            story.append(Paragraph(f"<i>Explanation: {explanation}</i>", analogy_style))
            story.append(Spacer(1, 6))
        story.append(Spacer(1, 6))

    # 5. FLASHCARDS (Q/A)
    if flashcards_data:
        story.append(Paragraph("📇 Flashcard Study Sheet (Q&A)", h2_style))
        card_table_data = [["#", "Question / Prompt", "Answer / Explanation"]]
        for c in flashcards_data:
            c_id = str(c.get("id", ""))
            c_q = Paragraph(clean_for_pdf(c.get("question", "")), body_style)
            c_a = Paragraph(clean_for_pdf(c.get("answer", "")), body_style)
            card_table_data.append([c_id, c_q, c_a])

        fc_table = Table(card_table_data, colWidths=[0.4 * inch, 2.8 * inch, 4.0 * inch])
        fc_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BG),
            ("TEXTCOLOR", (0, 0), (-1, 0), PRIMARY_BLUE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(fc_table)
        story.append(Spacer(1, 12))

    # 6. SENTIMENT & KEYWORDS
    story.append(Paragraph("📊 Sentiment & Keyword Analytics", h2_style))
    pol = nlp_data.get("polarity", 0.0)
    sub = nlp_data.get("subjectivity", 0.0)
    s_label = nlp_data.get("sentiment_label", "Neutral")
    sub_label = nlp_data.get("subjectivity_label", "Objective")

    sentiment_p = f"<b>Sentiment Polarity:</b> {pol:+.3f} ({s_label}) | <b>Subjectivity:</b> {sub:.3f} ({sub_label})"
    story.append(Paragraph(sentiment_p, body_style))

    kws = nlp_data.get("keywords", [])
    if kws:
        kw_str = ", ".join([f"<b>{clean_for_pdf(k)}</b> ({v})" for k, v in kws[:12]])
        story.append(Paragraph(f"<b>Top Keywords:</b> {kw_str}", body_style))
    story.append(Spacer(1, 10))

    # 7. WORD CLOUD IMAGE EMBED
    if wordcloud_buf:
        try:
            wordcloud_buf.seek(0)
            story.append(Paragraph("☁️ Visual Word Cloud", h2_style))
            img = Image(wordcloud_buf, width=6.8 * inch, height=3.4 * inch)
            story.append(img)
            story.append(Spacer(1, 10))
        except Exception:
            pass

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer
