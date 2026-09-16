# 🧠 AI Document Analyzer

A production-ready **AI Document Analyzer** web application featuring a modern **White and Blue** interface. Works **100% without an API key** using built-in intelligent NLP algorithms, with an optional toggle for LLM engines. 

Easily deployable to **Render** and structured as a clean **GitHub repository**.

---

## ✨ Features Checklist

- 📁 **Multi-Format Ingestion**: Drag & drop support for **PDF**, **DOCX**, and **TXT** files (up to 200MB).
- ✅ **Summaries**:
  - Executive Summary condensing the document's core thesis.
  - 5–7 Key Takeaways formatted as highlighted bullet points.
  - Word count, sentence count, estimated reading time, and readability grade level.
- ✅ **Teacher-Style Explanations**:
  - 💡 **The Big Picture (ELI5)**: Accessible explanation in plain conversational English.
  - 🔍 **Core Concepts Demystified**: Key terms broken down with concrete real-world analogies (*"Think of it like..."*), plain-English definitions, and practical significance.
  - 🪜 **Step-by-Step Learning Walkthrough**: Multi-phase conceptual learning guide.
  - 🎓 **Teacher's Study Tips**: Actionable advice for active recall.
- ✅ **Quiz Questions (MCQ)**:
  - 5 multiple-choice questions (A, B, C, D) generated from key facts and definitions.
  - **Interactive Quiz Mode**: Select answers, submit quiz, get instant score (e.g. `Score: 5/5 (100%)`), celebration effects, and color-coded answer feedback with full explanations.
- ✅ **Flashcards (Q/A)**:
  - Interactive flashcard study viewer with smooth *"Flip to Reveal Answer"* and *Next / Previous* navigation.
  - Full study sheet table view for quick review.
- 📊 **Sentiment + Keywords + Word Cloud**:
  - Sentiment polarity (`-1.0` to `+1.0`) and subjectivity gauge with status badge.
  - Top 12 keywords displayed in an interactive frequency bar chart.
  - High-resolution visual **Word Cloud** rendered in a royal blue palette matching the site's design.
- 📄 **Downloadable PDF Report**:
  - 1-click generation of a formatted **White & Royal Blue Executive PDF Report** via ReportLab.
  - Contains summaries, teacher breakdowns, quiz questions with answer key, flashcard tables, sentiment metrics, and the embedded word cloud image.

---
🌐 Live
---
**Link** 
- https://ai-document-analyzer-q4tg.onrender.com


## 🚀 Local Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py


