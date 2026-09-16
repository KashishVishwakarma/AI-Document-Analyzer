"""
modules/ai_engine.py
Zero-API-Key Intelligent Offline AI Engine for Document Analysis.
Provides Summaries, Teacher-Style Explanations, MCQ Quizzes, Flashcards,
Sentiment Analysis, Keyword Extraction, and Word Cloud Generation.
"""

import re
import random
from collections import Counter
from io import BytesIO
from typing import Dict, List, Tuple, Any

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize
except ImportError:
    nltk = None
    stopwords = None

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None

try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import google.generativeai as genai
except ImportError:
    genai = None


DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "him", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my", "no", "nor",
    "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "out",
    "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
    "also", "etc", "e.g", "i.e", "one", "two", "three", "four", "first", "second",
    "introduction", "overview", "conclusion", "section", "chapter", "table", "figure"
}


def get_stopwords() -> set:
    if stopwords:
        try:
            return set(stopwords.words("english")).union(DEFAULT_STOPWORDS)
        except Exception:
            pass
    return DEFAULT_STOPWORDS


def split_sentences(text: str) -> List[str]:
    """Splits text into meaningful complete sentences across paragraphs, removing bare headers."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    valid = []
    for p in paragraphs:
        raw_sents = []
        if nltk:
            try:
                raw_sents = sent_tokenize(p)
            except Exception:
                raw_sents = re.split(r"(?<=[.!?])\s+", p)
        else:
            raw_sents = re.split(r"(?<=[.!?])\s+", p)

        for s in raw_sents:
            s_clean = s.strip()
            s_clean = re.sub(r"^\d+[\.\)]\s*", "", s_clean)
            if len(s_clean.split()) >= 6 and re.search(r"[a-zA-Z]", s_clean):
                valid.append(s_clean)
    return valid


def extract_words(text: str, remove_stopwords: bool = True) -> List[str]:
    words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    if remove_stopwords:
        sw = get_stopwords()
        return [w for w in words if w not in sw]
    return words


# 1. SUMMARIES (NO API KEY)
def generate_summary(text: str, top_k: int = 5) -> Dict[str, Any]:
    sentences = split_sentences(text)
    if not sentences:
        return {
            "executive_summary": "No sufficient text available to summarize.",
            "key_takeaways": []
        }

    words = extract_words(text, remove_stopwords=True)
    if not words:
        return {
            "executive_summary": sentences[0] if sentences else "",
            "key_takeaways": sentences[:3]
        }

    word_freq = Counter(words)
    max_freq = max(word_freq.values()) if word_freq else 1
    freq_weights = {w: c / max_freq for w, c in word_freq.items()}

    sentence_scores = []
    total_sents = len(sentences)

    for i, sent in enumerate(sentences):
        sent_words = extract_words(sent, remove_stopwords=True)
        if len(sent_words) < 3:
            continue
        
        score = sum(freq_weights.get(w, 0) for w in sent_words) / (len(sent_words) ** 0.5)

        if i == 0:
            score *= 1.4
        elif i < 3:
            score *= 1.2
        elif i > total_sents - 3:
            score *= 1.15

        clue_words = {"demonstrates", "concludes", "results", "crucial", "essential", "primary", "important", "found", "shows", "significant", "enables", "refers"}
        if any(w in clue_words for w in sent_words):
            score *= 1.25

        sentence_scores.append((score, i, sent))

    sentence_scores.sort(key=lambda x: x[0], reverse=True)

    selected_indices = sorted([item[1] for item in sentence_scores[:top_k]])
    executive_summary_sentences = [sentences[idx] for idx in selected_indices]
    executive_summary = " ".join(executive_summary_sentences)

    takeaways_indices = sorted([item[1] for item in sentence_scores[:min(7, len(sentence_scores))]])
    takeaways = []
    for idx in takeaways_indices:
        s = sentences[idx].strip()
        if not s.endswith(('.', '!', '?')):
            s += '.'
        s = s[0].upper() + s[1:]
        takeaways.append(s)

    return {
        "executive_summary": executive_summary,
        "key_takeaways": takeaways
    }


# 2. TEACHER-STYLE EXPLANATION (NO API KEY)
def generate_teacher_explanation(text: str) -> Dict[str, Any]:
    sentences = split_sentences(text)
    words = extract_words(text, remove_stopwords=True)
    word_freq = Counter(words)

    entities = re.findall(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\b", text)
    filtered_entities = [
        e for e in entities 
        if e.lower() not in DEFAULT_STOPWORDS 
        and not any(sw in e.lower().split() for sw in ["introduction", "chapter", "section", "table", "figure", "page"])
        and len(e.split()) <= 3
    ]
    entity_counts = Counter(filtered_entities)

    top_concepts = [e for e, c in entity_counts.most_common(10)]
    if len(top_concepts) < 4:
        top_concepts += [w.title() for w, c in word_freq.most_common(8) if w.title() not in top_concepts]

    top_concepts = top_concepts[:5]

    analogy_bank = {
        "intelligence": "Like a digital brain that can observe patterns and solve problems without tiring.",
        "learning": "Like practicing a musical instrument: making mistakes, adjusting technique, and sounding cleaner with each repeat.",
        "data": "Like flour and water for a baker; raw ingredients that must be carefully prepared to create bread.",
        "model": "Like an architect's detailed blueprint capturing the rules of how a building stands.",
        "network": "Like a company where information passes through specialized departments before reaching the final decision.",
        "neural": "Like interconnected runners passing a baton forward in a stadium relay.",
        "training": "Like a student completing practice exams and reviewing the answer key to improve.",
        "algorithm": "Like a foolproof recipe for baking a cake with exact step-by-step measurements.",
        "supervised": "Like studying with a private tutor who marks every answer right or wrong immediately.",
        "unsupervised": "Like organizing your messy closet into neat piles based on color and style without instructions.",
        "reinforcement": "Like training a puppy with treats: helpful actions earn rewards, unwanted actions get ignored.",
        "classification": "Like sorting incoming physical mail into bills, letters, and advertisements.",
        "regression": "Like estimating the price of a house based on square footage, location, and condition.",
        "transformer": "Like a master linguist who understands an entire book in context rather than reading word by word."
    }

    first_few = " ".join(sentences[:2]) if len(sentences) >= 2 else (sentences[0] if sentences else text[:200])
    main_subject = top_concepts[0] if top_concepts else "this subject"
    big_picture = (
        f"Imagine you are explaining this document to a curious student: "
        f"At its core, this document explores **{main_subject}**. "
        f"It shows how foundational principles connect together to solve real-world problems. Specifically, {first_few}"
    )

    concept_breakdowns = []
    for concept in top_concepts:
        matched_sentences = [s for s in sentences if concept.lower() in s.lower()]
        definition = matched_sentences[0] if matched_sentences else f"A foundational pillar of the document's subject matter: {concept}."
        
        matched_analogy = None
        for key, ana in analogy_bank.items():
            if key in concept.lower():
                matched_analogy = ana
                break
        if not matched_analogy:
            matched_analogy = f"Think of {concept} like an essential tool in a master craftsman's workshop—specialized for its specific duty."

        concept_breakdowns.append({
            "concept": concept,
            "plain_english": definition,
            "analogy": matched_analogy,
            "why_it_matters": f"Understanding {concept} gives you the clarity to see how the overall system works together."
        })

    milestones = []
    step_chunks = [sentences[i:i + max(1, len(sentences) // 4)] for i in range(0, len(sentences), max(1, len(sentences) // 4))]
    step_titles = [
        "Phase 1: Setting the Foundation & Core Definitions",
        "Phase 2: Exploring Key Mechanisms & Methodologies",
        "Phase 3: Deepening Architectural & Practical Applications",
        "Phase 4: Synthesis, Implications & Future Horizons"
    ]
    for i, chunk in enumerate(step_chunks[:4]):
        title = step_titles[i] if i < len(step_titles) else f"Phase {i+1}: Advanced Insights"
        summary_chunk = chunk[0] if chunk else "Core conceptual overview."
        milestones.append({
            "step_title": title,
            "description": summary_chunk
        })

    study_tips = [
        "Focus on the relationships between key concepts rather than trying to memorize isolated phrases.",
        "Try re-explaining one concept out loud using the real-world analogies provided above.",
        "Test your active recall with the interactive quiz below to solidify your comprehension."
    ]

    return {
        "big_picture": big_picture,
        "concepts": concept_breakdowns,
        "milestones": milestones,
        "study_tips": study_tips
    }


# 3. QUIZ QUESTIONS (MCQ) (NO API KEY)
def generate_mcq_quiz(text: str, num_questions: int = 5) -> List[Dict[str, Any]]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    entities = re.findall(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\b", text)
    candidate_distractors = list(set([
        e for e in entities 
        if len(e.split()) <= 3 
        and e.lower() not in DEFAULT_STOPWORDS
        and not any(sw in e.lower() for sw in ["introduction", "chapter", "section", "table", "figure"])
    ]))

    default_distractors = [
        "Supervised Learning", "Unsupervised Learning", "Reinforcement Learning",
        "Deep Learning", "Neural Networks", "Convolutional Networks", "Transformers",
        "Feature Engineering", "Gradient Descent", "Linear Regression", "Decision Trees"
    ]
    candidate_distractors.extend(default_distractors)
    candidate_distractors = list(dict.fromkeys(candidate_distractors))

    patterns = [
        (r"^(.+?)\s+(?:refers to|is defined as|is a|means|consists of)\s+(.+)$", "definition"),
        (r"^(.+?)\s+(?:enables|provides|allows|produces|aims to)\s+(.+)$", "function"),
        (r"^(.+?)\s+(?:includes|contains|comprises)\s+(.+)$", "inclusion"),
    ]

    mcqs = []
    used_sentences = set()

    for sent in sentences:
        if len(sent.split()) < 8 or len(sent.split()) > 40:
            continue
        if sent in used_sentences:
            continue

        for pat, ptype in patterns:
            m = re.search(pat, sent, re.IGNORECASE)
            if m:
                subject = m.group(1).strip()
                predicate = m.group(2).strip()
                clean_subject = re.sub(r"^(The|A|An|In general,|Furthermore,|Notably,)\s+", "", subject, flags=re.IGNORECASE).strip()
                
                if 2 <= len(clean_subject.split()) <= 5 and not clean_subject.lower().startswith(("it", "they", "this", "these", "there")):
                    question = f"According to the document, what matches this description:\n\"{predicate[:150]}...\"?"
                    correct_answer = clean_subject

                    distractors = [d for d in candidate_distractors if d.lower() != correct_answer.lower() and d.lower() not in predicate.lower()]
                    random.seed(len(sent) + len(mcqs))
                    if len(distractors) < 3:
                        distractors += ["Support Vector Machines", "K-Means Clustering", "Random Forests"]
                    
                    selected_distractors = random.sample(distractors, 3)
                    options = [correct_answer] + selected_distractors
                    random.shuffle(options)
                    
                    letters = ["A", "B", "C", "D"]
                    correct_idx = options.index(correct_answer)
                    correct_letter = letters[correct_idx]
                    formatted_options = [f"{letters[idx]}) {opt}" for idx, opt in enumerate(options)]

                    mcqs.append({
                        "id": len(mcqs) + 1,
                        "question": question,
                        "options": formatted_options,
                        "raw_options": options,
                        "correct": correct_letter,
                        "correct_answer": correct_answer,
                        "explanation": f"In the document: '{sent}'"
                    })
                    used_sentences.add(sent)
                    break
            if len(mcqs) >= num_questions:
                break
        if len(mcqs) >= num_questions:
            break

    if len(mcqs) < num_questions:
        candidate_sents = [
            s for s in sentences 
            if s not in used_sentences 
            and 12 <= len(s.split()) <= 35
            and not any(sw in s.lower() for sw in ["introduction", "chapter", "section"])
        ]
        for sent in candidate_sents:
            words_in_sent = [w for w in extract_words(sent) if len(w) > 4 and w.lower() not in DEFAULT_STOPWORDS]
            if words_in_sent:
                target_word = words_in_sent[0].title()
                blanked_sent = re.sub(r"\b" + re.escape(target_word) + r"\b", "_______", sent, count=1, flags=re.IGNORECASE)
                if "_______" in blanked_sent:
                    question = f"Fill in the blank based on the document:\n\"{blanked_sent}\""
                    correct_answer = target_word

                    distractors = [
                        w.title() for w in extract_words(text) 
                        if w.lower() != target_word.lower() 
                        and len(w) > 4 
                        and w.lower() not in DEFAULT_STOPWORDS
                    ]
                    distractors = list(dict.fromkeys(distractors))
                    if len(distractors) < 3:
                        distractors += ["Optimization", "Architecture", "Regulation", "Framework"]

                    random.seed(len(sent))
                    selected_distractors = random.sample(distractors[:15], 3)
                    options = [correct_answer] + selected_distractors
                    random.shuffle(options)

                    letters = ["A", "B", "C", "D"]
                    correct_idx = options.index(correct_answer)
                    correct_letter = letters[correct_idx]
                    formatted_options = [f"{letters[idx]}) {opt}" for idx, opt in enumerate(options)]

                    mcqs.append({
                        "id": len(mcqs) + 1,
                        "question": question,
                        "options": formatted_options,
                        "raw_options": options,
                        "correct": correct_letter,
                        "correct_answer": correct_answer,
                        "explanation": f"Full statement: '{sent}'"
                    })
                    used_sentences.add(sent)
                    if len(mcqs) >= num_questions:
                        break

    return mcqs[:num_questions]


# 4. FLASHCARDS (Q/A) (NO API KEY)
def generate_flashcards(text: str, num_cards: int = 8) -> List[Dict[str, str]]:
    sentences = split_sentences(text)
    flashcards = []
    used_sents = set()

    for sent in sentences:
        if len(sent.split()) < 6 or sent in used_sents:
            continue

        match = re.search(r"^(.*?)\s+(?:refers to|is defined as|is a|means|consists of)\s+(.+)$", sent, re.IGNORECASE)
        if match:
            term = re.sub(r"^(The|A|An)\s+", "", match.group(1).strip(), flags=re.IGNORECASE)
            definition = match.group(2).strip()
            if 1 <= len(term.split()) <= 4 and not term.lower().startswith(("it", "this", "they")):
                flashcards.append({
                    "id": len(flashcards) + 1,
                    "question": f"What is {term}?",
                    "answer": f"{term} refers to {definition}"
                })
                used_sents.add(sent)

        elif ":" in sent:
            parts = sent.split(":", 1)
            term = parts[0].strip()
            explanation = parts[1].strip()
            term = re.sub(r"^\d+[\.\)]\s*", "", term)
            if 1 <= len(term.split()) <= 5 and len(explanation.split()) >= 4:
                flashcards.append({
                    "id": len(flashcards) + 1,
                    "question": f"Define: {term}",
                    "answer": explanation
                })
                used_sents.add(sent)

        if len(flashcards) >= num_cards:
            break

    if len(flashcards) < num_cards:
        for sent in sentences:
            if sent not in used_sents and 8 <= len(sent.split()) <= 30:
                flashcards.append({
                    "id": len(flashcards) + 1,
                    "question": f"Key Principle #{len(flashcards)+1}:",
                    "answer": sent
                })
                used_sents.add(sent)
            if len(flashcards) >= num_cards:
                break

    return flashcards[:num_cards]


# 5. SENTIMENT, KEYWORDS & ENTITIES
def analyze_sentiment_and_keywords(text: str) -> Dict[str, Any]:
    polarity = 0.0
    subjectivity = 0.5
    if TextBlob:
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
        except Exception:
            pass

    if polarity >= 0.35:
        sentiment_label = "Strongly Positive / Optimistic"
        sentiment_color = "#10B981"
    elif polarity >= 0.08:
        sentiment_label = "Positive / Constructive"
        sentiment_color = "#3B82F6"
    elif polarity >= -0.08:
        sentiment_label = "Neutral / Objective"
        sentiment_color = "#64748B"
    elif polarity >= -0.35:
        sentiment_label = "Cautious / Critical"
        sentiment_color = "#F59E0B"
    else:
        sentiment_label = "Negative / Problematic"
        sentiment_color = "#EF4444"

    if subjectivity >= 0.6:
        subjectivity_label = "Opinionated / Evaluative"
    elif subjectivity >= 0.3:
        subjectivity_label = "Balanced (Facts & Analysis)"
    else:
        subjectivity_label = "Factual & Objective"

    words = extract_words(text, remove_stopwords=True)
    word_freq = Counter(words)
    top_keywords = word_freq.most_common(12)

    entities = re.findall(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\b", text)
    valid_entities = [
        e for e in entities 
        if e.lower() not in DEFAULT_STOPWORDS 
        and len(e) > 2
        and not any(sw in e.lower() for sw in ["introduction", "chapter", "section"])
    ]
    top_entities = [item[0] for item in Counter(valid_entities).most_common(15)]

    return {
        "polarity": round(polarity, 3),
        "subjectivity": round(subjectivity, 3),
        "sentiment_label": sentiment_label,
        "sentiment_color": sentiment_color,
        "subjectivity_label": subjectivity_label,
        "keywords": top_keywords,
        "entities": top_entities,
        "word_list": words
    }


# 6. WORD CLOUD (BLUE & WHITE PALETTE)
def generate_wordcloud_image(words: List[str]) -> BytesIO | None:
    if not words or not WordCloud:
        return None

    try:
        wc = WordCloud(
            width=1000,
            height=500,
            background_color="#FFFFFF",
            colormap="Blues",
            max_words=120,
            contour_width=1,
            contour_color="#93C5FD",
            prefer_horizontal=0.85
        ).generate(" ".join(words))

        fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig.tight_layout(pad=0)

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#FFFFFF")
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception:
        return None
