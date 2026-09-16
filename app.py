"""
app.py
AI Document Analyzer - Production Streamlit Web Application.
Zero API Key Required • Deployable on Render • Modern White & Blue UI.
"""

import os
import io
import time
from typing import Dict, Any, List

import streamlit as st
import pandas as pd

from modules.text_extractor import extract_text_from_file, calculate_document_stats
from modules.ai_engine import (
    generate_summary,
    generate_teacher_explanation,
    generate_mcq_quiz,
    generate_flashcards,
    analyze_sentiment_and_keywords,
    generate_wordcloud_image,
    genai
)
from modules.pdf_report import generate_pdf_report


st.set_page_config(
    page_title="AI Document Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
.stApp {
    background-color: #F8FAFC;
    color: #0F172A;
}

.hero-container {
    background: linear-gradient(135deg, #1E40AF 0%, #2563EB 50%, #3B82F6 100%);
    border-radius: 16px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 2rem;
    color: white;
    box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.25);
}

.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    color: #FFFFFF !important;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.hero-subtitle {
    font-size: 1.15rem;
    color: #DBEAFE !important;
    margin-bottom: 1.2rem;
    font-weight: 500;
}

.badge-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: 0.8rem;
}

.feature-pill {
    background: rgba(255, 255, 255, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.35);
    backdrop-filter: blur(8px);
    color: #FFFFFF;
    padding: 0.4rem 0.9rem;
    border-radius: 9999px;
    font-size: 0.9rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
}

.white-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
}

.concept-card {
    background: #FFFFFF;
    border-left: 4px solid #2563EB;
    border-top: 1px solid #E2E8F0;
    border-right: 1px solid #E2E8F0;
    border-bottom: 1px solid #E2E8F0;
    border-radius: 0 12px 12px 0;
    padding: 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
}

.analogy-box {
    background-color: #EFF6FF;
    border: 1px dashed #93C5FD;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-top: 0.6rem;
    color: #1E40AF;
    font-style: italic;
}

div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
}

div[data-testid="stMetricLabel"] {
    color: #475569 !important;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #1E40AF !important;
    font-weight: 800;
}

.stButton > button {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
    color: white;
    font-weight: 600;
    border-radius: 8px;
    border: none;
    padding: 0.55rem 1.25rem;
    box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
    box-shadow: 0 6px 12px -1px rgba(37, 99, 235, 0.35);
    transform: translateY(-1px);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 2px solid #E2E8F0;
    padding-bottom: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
    color: #475569;
    background-color: transparent;
}

.stTabs [aria-selected="true"] {
    background-color: #EFF6FF !important;
    color: #1D4ED8 !important;
}

.flashcard-display {
    background: linear-gradient(135deg, #FFFFFF 0%, #F0F7FF 100%);
    border: 2px solid #93C5FD;
    border-radius: 16px;
    padding: 2.5rem;
    min-height: 220px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.1);
    margin: 1.5rem 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="hero-container">
    <div class="hero-title">🧠 AI Document Analyzer</div>
    <div class="hero-subtitle">Upload a <b>PDF / DOCX / TXT</b> document and instantly receive:</div>
    <div class="badge-grid">
        <span class="feature-pill">✅ Summaries</span>
        <span class="feature-pill">✅ Teacher-style explanations</span>
        <span class="feature-pill">✅ Quiz questions (MCQ)</span>
        <span class="feature-pill">✅ Flashcards (Q/A)</span>
        <span class="feature-pill">📊 Sentiment + keywords + word cloud</span>
        <span class="feature-pill">📄 Downloadable PDF report</span>
        <span class="feature-pill">⚡ 100% Free & No API Key Required</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚙️ Engine Settings")
    st.caption("AI Document Analyzer v2.0 • Offline AI Mode")

    engine_mode = st.radio(
        "Select AI Processing Mode:",
        ["⚡ Smart Offline AI (No API Key Required)", "🤖 Google Gemini (Optional API Key)"],
        index=0,
        help="The Smart Offline AI engine runs locally with zero API key, zero cost, and instant speed!"
    )

    gemini_key = ""
    gemini_model_name = "gemini-2.0-flash"

    if "Google Gemini" in engine_mode:
        gemini_key = st.text_input(
            "Gemini API Key:",
            type="password",
            help="Enter your Google Gemini API key if you wish to use Gemini LLM."
        )
        gemini_model_name = st.selectbox(
            "Gemini Model:",
            ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        )
        if not gemini_key:
            st.info("💡 No key provided. Falling back to the Smart Offline AI Engine automatically.")

    st.markdown("---")
    st.markdown("### 📁 Try with Sample Document")
    sample_file_path = os.path.join(os.path.dirname(__file__), "sample_docs", "sample_ai_overview.txt")
    if os.path.exists(sample_file_path):
        if st.button("📄 Load Sample AI Document"):
            with open(sample_file_path, "r", encoding="utf-8") as f:
                st.session_state["sample_text"] = f.read()
                st.session_state["sample_name"] = "sample_ai_overview.txt"
            st.success("Sample document loaded! Scroll down to view analysis.")

    st.markdown("---")
    st.markdown("### 🚀 Deployment Info")
    st.markdown(
        "- **Platform:** Ready for [Render](https://render.com)\n"
        "- **RAM Footprint:** < 250 MB (Free tier safe)\n"
        "- **API Key Needed:** **None** (100% Offline)\n"
        "- **Formats:** PDF, DOCX, TXT up to 200MB"
    )

st.markdown("### 📁 Upload PDF / DOCX / TXT")

uploaded_file = st.file_uploader(
    "Drag and drop file here",
    type=["pdf", "docx", "txt"],
    help="Limit 200MB per file • PDF, DOCX, TXT"
)

document_text = ""
document_name = ""

if uploaded_file is not None:
    document_name = uploaded_file.name
    with st.spinner("📥 Extracting document content..."):
        text, err = extract_text_from_file(uploaded_file)
        if err:
            st.error(f"Error reading file: {err}")
        else:
            document_text = text
elif "sample_text" in st.session_state and st.session_state["sample_text"]:
    document_text = st.session_state["sample_text"]
    document_name = st.session_state.get("sample_name", "sample_ai_overview.txt")
    st.info(f"✨ Currently analyzing loaded sample file: **{document_name}**")

if document_text:
    if len(document_text.strip()) < 80:
        st.warning("⚠️ The uploaded document contains very little readable text (less than 80 characters). Please upload a document with more textual content.")
        st.stop()

    stats = calculate_document_stats(document_text)

    st.markdown("---")
    st.markdown("### 📊 Document Overview")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Words", f"{stats['word_count']:,}")
    with m2:
        st.metric("Estimated Read Time", f"{stats['reading_time_min']} min")
    with m3:
        st.metric("Total Sentences", f"{stats['sentence_count']}")
    with m4:
        st.metric("Readability Level", stats['readability_level'])

    with st.expander("👁️ View Extracted Text Preview", expanded=False):
        st.text_area(
            "Extracted Content (first 2,000 characters)",
            document_text[:2000] + ("..." if len(document_text) > 2000 else ""),
            height=200,
            disabled=True
        )

    with st.spinner("⚡ Running AI Document Analysis (Summaries, Teacher Explanations, MCQs, Flashcards, Analytics)..."):
        summary_data = generate_summary(document_text, top_k=5)
        teacher_data = generate_teacher_explanation(document_text)
        quiz_data = generate_mcq_quiz(document_text, num_questions=5)
        flashcards_data = generate_flashcards(document_text, num_cards=8)
        nlp_data = analyze_sentiment_and_keywords(document_text)
        wc_buf = generate_wordcloud_image(nlp_data.get("word_list", []))

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📑 Summaries",
        "👨‍🏫 Teacher Mode",
        "🎯 Quiz Questions (MCQ)",
        "📇 Flashcards (Q/A)",
        "📊 Sentiment & Word Cloud",
        "📄 Downloadable PDF Report"
    ])

    with tab1:
        st.markdown("#### 📑 Executive Summary")
        st.markdown(f"""
        <div class="white-card">
            <p style="font-size: 1.05rem; line-height: 1.7; color: #1E293B;">
                {summary_data.get('executive_summary', 'No summary generated.')}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🔑 Key Takeaways")
        takeaways = summary_data.get("key_takeaways", [])
        if takeaways:
            for idx, t in enumerate(takeaways, 1):
                st.markdown(f"""
                <div style="background: #FFFFFF; border-left: 4px solid #2563EB; border-radius: 8px; padding: 12px 18px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                    <span style="font-weight: 700; color: #2563EB; margin-right: 8px;">#{idx}</span>
                    <span style="color: #1E293B; font-size: 0.98rem;">{t}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No takeaways extracted.")

    with tab2:
        st.markdown("#### 💡 The Big Picture (Explain Like I'm 5)")
        st.markdown(f"""
        <div class="white-card" style="border-left: 6px solid #3B82F6;">
            <p style="font-size: 1.05rem; line-height: 1.6; color: #1E293B;">
                {teacher_data.get('big_picture', '')}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🔍 Core Concepts Demystified")
        concepts = teacher_data.get("concepts", [])
        if concepts:
            for c in concepts:
                st.markdown(f"""
                <div class="concept-card">
                    <div style="font-size: 1.15rem; font-weight: 700; color: #1E3A8A; margin-bottom: 6px;">
                        📌 {c.get('concept')}
                    </div>
                    <div style="color: #334155; font-size: 0.95rem; line-height: 1.5;">
                        <b>In Plain English:</b> {c.get('plain_english')}
                    </div>
                    <div class="analogy-box">
                        <b>💡 Analogy:</b> {c.get('analogy')}
                    </div>
                    <div style="color: #64748B; font-size: 0.85rem; margin-top: 6px;">
                        <b>Why it matters:</b> {c.get('why_it_matters')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("#### 🪜 Step-by-Step Learning Walkthrough")
        milestones = teacher_data.get("milestones", [])
        for m in milestones:
            with st.expander(f"📘 {m.get('step_title')}", expanded=True):
                st.write(m.get('description'))

        st.markdown("#### 🎓 Teacher's Study Takeaways")
        tips = teacher_data.get("study_tips", [])
        for tip in tips:
            st.markdown(f"- 🌟 {tip}")

    with tab3:
        st.markdown("#### 🎯 Interactive Document Quiz")
        st.caption("Test your understanding! Select your answers below and click **Submit Quiz** to check your score.")

        if not quiz_data:
            st.info("No quiz questions could be automatically formed from this text.")
        else:
            if "quiz_answers" not in st.session_state:
                st.session_state["quiz_answers"] = {}
            if "quiz_submitted" not in st.session_state:
                st.session_state["quiz_submitted"] = False

            user_choices = {}

            for q in quiz_data:
                q_id = q["id"]
                st.markdown(f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px; margin-bottom: 12px;">
                    <div style="font-weight: 700; color: #1E3A8A; font-size: 1.05rem; margin-bottom: 8px;">
                        Question {q_id}: {q['question']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                user_choices[q_id] = st.radio(
                    f"Select Answer for Question {q_id}:",
                    options=q["options"],
                    key=f"q_radio_{q_id}",
                    label_visibility="collapsed"
                )

            col_submit, col_reset = st.columns([1, 4])
            with col_submit:
                if st.button("🚀 Submit Quiz"):
                    st.session_state["quiz_submitted"] = True
            with col_reset:
                if st.button("🔄 Reset Quiz"):
                    st.session_state["quiz_submitted"] = False
                    st.rerun()

            if st.session_state.get("quiz_submitted", False):
                st.markdown("---")
                score = 0
                total = len(quiz_data)

                for q in quiz_data:
                    q_id = q["id"]
                    chosen = user_choices.get(q_id, "")
                    chosen_letter = chosen[:1] if chosen else ""
                    is_correct = (chosen_letter == q["correct"])

                    if is_correct:
                        score += 1
                        st.success(f"✅ **Question {q_id}: Correct!** You selected {chosen}. \n\n*{q['explanation']}*")
                    else:
                        correct_opt_str = next((opt for opt in q["options"] if opt.startswith(q["correct"])), q["correct"])
                        st.error(f"❌ **Question {q_id}: Incorrect.** Your choice: {chosen}. \n\n**Correct Answer:** {correct_opt_str}\n\n*{q['explanation']}*")

                pct = int((score / total) * 100)
                st.markdown(f"### 🏆 Your Quiz Score: **{score} / {total}** ({pct}%)")
                if score == total:
                    st.balloons()
                    st.success("🎉 Perfect Score! Excellent mastery of the document content!")
                elif score >= total / 2:
                    st.info("👍 Great effort! Review the explanations above to achieve a perfect score.")
                else:
                    st.warning("Keep practicing! Review the Teacher Mode tab to strengthen your understanding.")

    with tab4:
        st.markdown("#### 📇 Interactive Flashcards")
        st.caption("Flip through key concepts and definitions to test your active recall.")

        if not flashcards_data:
            st.info("No flashcards available for this document.")
        else:
            if "fc_index" not in st.session_state:
                st.session_state["fc_index"] = 0
            if "fc_revealed" not in st.session_state:
                st.session_state["fc_revealed"] = False

            current_idx = st.session_state["fc_index"]
            total_cards = len(flashcards_data)
            current_card = flashcards_data[current_idx]

            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.progress((current_idx + 1) / total_cards)
                st.markdown(f"<p style='text-align: center; color: #475569; font-weight: 600;'>Card {current_idx + 1} of {total_cards}</p>", unsafe_allow_html=True)

            is_rev = st.session_state["fc_revealed"]
            card_content = current_card["answer"] if is_rev else current_card["question"]
            card_badge = "💡 ANSWER / EXPLANATION" if is_rev else "❓ QUESTION / PROMPT"
            badge_color = "#10B981" if is_rev else "#2563EB"

            st.markdown(f"""
            <div class="flashcard-display">
                <span style="background: {badge_color}; color: white; padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; font-weight: 700; margin-bottom: 12px;">
                    {card_badge}
                </span>
                <div style="font-size: 1.3rem; font-weight: 600; color: #0F172A; max-width: 800px; line-height: 1.6;">
                    {card_content}
                </div>
            </div>
            """, unsafe_allow_html=True)

            nav1, nav2, nav3 = st.columns([1, 2, 1])
            with nav1:
                if st.button("⬅️ Previous Card", disabled=(current_idx == 0)):
                    st.session_state["fc_index"] -= 1
                    st.session_state["fc_revealed"] = False
                    st.rerun()
            with nav2:
                flip_label = "👁️ Hide Answer (Show Question)" if is_rev else "🔍 Reveal Answer / Definition"
                if st.button(flip_label, use_container_width=True):
                    st.session_state["fc_revealed"] = not is_rev
                    st.rerun()
            with nav3:
                if st.button("Next Card ➡️", disabled=(current_idx == total_cards - 1)):
                    st.session_state["fc_index"] += 1
                    st.session_state["fc_revealed"] = False
                    st.rerun()

            with st.expander("📋 View All Flashcards (Study Sheet)", expanded=False):
                fc_df = pd.DataFrame(flashcards_data)
                st.dataframe(fc_df[["id", "question", "answer"]], use_container_width=True, hide_index=True)

    with tab5:
        st.markdown("#### 📊 Sentiment & Linguistic Analysis")
        sc1, sc2 = st.columns(2)
        with sc1:
            pol = nlp_data.get("polarity", 0.0)
            st.metric(
                "😊 Sentiment Polarity",
                f"{pol:+.3f}",
                delta=nlp_data.get("sentiment_label", "Neutral"),
                help="-1.0 (strongly negative) to +1.0 (strongly positive)"
            )
        with sc2:
            sub = nlp_data.get("subjectivity", 0.0)
            st.metric(
                "🎯 Subjectivity Score",
                f"{sub:.3f}",
                delta=nlp_data.get("subjectivity_label", "Factual"),
                help="0.0 (completely objective/factual) to 1.0 (highly subjective/opinionated)"
            )

        st.markdown("#### 🔑 Top Keywords & Frequency")
        kws = nlp_data.get("keywords", [])
        if kws:
            kw_df = pd.DataFrame(kws, columns=["Keyword", "Occurrences"])
            st.bar_chart(kw_df.set_index("Keyword"), color="#2563EB")
        else:
            st.info("No prominent keywords extracted.")

        st.markdown("#### ☁️ Visual Word Cloud (Blue Palette)")
        if wc_buf:
            st.image(wc_buf, caption="Word frequency distribution rendered with royal blue palette", use_container_width=True)
        else:
            st.info("Insufficient words to render word cloud.")

    with tab6:
        st.markdown("#### 📄 Executive PDF Report Generator")
        st.write("Generate and download a comprehensive, beautifully styled **White & Royal Blue PDF Report** containing all summaries, teacher explanations, quiz questions with answer key, flashcard study sheets, sentiment metrics, and embedded word cloud.")

        try:
            pdf_bytes = generate_pdf_report(
                filename=document_name or "Document_Analysis",
                stats=stats,
                summary_data=summary_data,
                teacher_data=teacher_data,
                quiz_data=quiz_data,
                flashcards_data=flashcards_data,
                nlp_data=nlp_data,
                wordcloud_buf=wc_buf
            )

            report_filename = f"AI_Document_Report_{os.path.splitext(document_name)[0]}.pdf" if document_name else "AI_Document_Report.pdf"

            st.download_button(
                label="📥 Download Executive PDF Report",
                data=pdf_bytes,
                file_name=report_filename,
                mime="application/pdf",
                help="Click to download the formatted PDF report"
            )
            st.success("✅ PDF report generated and ready for instant download!")
        except Exception as e:
            st.error(f"Could not generate PDF: {str(e)}")

else:
    st.info("💡 **Getting Started:** Upload a PDF, DOCX, or TXT document above, or click **'📄 Load Sample AI Document'** in the sidebar to test all features immediately!")
