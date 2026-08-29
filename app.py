"""
Multilingual NLP Studio
========================
Neural translation and sentiment analysis for multilingual text.

A Streamlit web application built on Hugging Face MarianMT (translation)
and DistilBERT (sentiment analysis). This is the web successor to a PyQt5
desktop prototype -- see README.md for the full writeup of what changed
and why.
"""

from __future__ import annotations

import streamlit as st

from src.evaluation import TEST_SAMPLES, run_evaluation
from src.sentiment import SentimentError, analyze_sentiment
from src.translation import (
    ENGLISH,
    SUPPORTED_LANGUAGES,
    InputTooLongError,
    ModelLoadError,
    TranslationError,
    UnsupportedLanguageError,
    translate_text,
)
from src.utils import timestamp, truncate

# NOTE on caching: src/translation.py and src/sentiment.py keep their own
# module-level dict caches (see _model_cache / _sentiment_pipeline). Because
# Streamlit runs your script inside a long-lived server process and re-runs
# the script body on each interaction WITHOUT reloading Python modules,
# that module-level cache already persists across reruns for free -- a
# model downloaded on the first translation stays in memory for the rest
# of the process's life. This keeps the NLP modules Streamlit-agnostic
# (they can be imported and tested with no Streamlit dependency at all)
# while still getting "load once, reuse everywhere" behavior in the app.

# ---------------------------------------------------------------------------
# Page config & global style
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multilingual NLP Studio",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    #MainMenu, footer {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    .studio-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.15rem;
    }
    .studio-subtitle {
        font-size: 1.02rem;
        color: #6b7280;
        margin-bottom: 1.6rem;
    }

    .nlp-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
    }
    .nlp-card h4 {
        margin-top: 0;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6b7280;
        margin-bottom: 0.75rem;
    }

    .sentiment-badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
    }
    .sentiment-positive {
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
    }
    .sentiment-negative {
        background: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fecaca;
    }

    .translation-note {
        font-size: 0.82rem;
        color: #9ca3af;
        margin-top: 0.5rem;
    }

    .metric-table td, .metric-table th {
        padding: 0.35rem 0.6rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts, most recent first


def add_history_entry(entry: dict) -> None:
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:25]  # cap history size


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌐 Multilingual NLP Studio")
    page = st.radio(
        "Navigate",
        ["Translate & Analyze", "Model Evaluation", "About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(
        "Translation: Helsinki-NLP MarianMT\n\n"
        "Sentiment: DistilBERT (SST-2, English-only)\n\n"
        "Models download on first use and are cached for the session."
    )

# ---------------------------------------------------------------------------
# Page: Translate & Analyze
# ---------------------------------------------------------------------------
if page == "Translate & Analyze":
    st.markdown('<div class="studio-title">Multilingual NLP Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="studio-subtitle">Neural translation and sentiment analysis '
        "for multilingual text.</div>",
        unsafe_allow_html=True,
    )

    col_source, col_target = st.columns(2)
    with col_source:
        source_lang = st.selectbox("Source language", SUPPORTED_LANGUAGES, index=0)
    with col_target:
        default_target_idx = 1 if len(SUPPORTED_LANGUAGES) > 1 else 0
        target_lang = st.selectbox("Target language", SUPPORTED_LANGUAGES, index=default_target_idx)

    input_text = st.text_area(
        "Input text",
        placeholder="Enter text to translate...",
        height=140,
    )

    translate_clicked = st.button("Translate", type="primary", use_container_width=False)

    if translate_clicked:
        translated = None
        sentiment_result = None
        translation_failed = False

        try:
            with st.spinner(f"Translating {source_lang} → {target_lang}..."):
                translated = translate_text(input_text, source_lang, target_lang)
        except (TranslationError, UnsupportedLanguageError, InputTooLongError, ModelLoadError) as exc:
            st.error(str(exc))
            translation_failed = True

        if not translation_failed:
            try:
                with st.spinner("Analyzing sentiment..."):
                    sentiment_result = analyze_sentiment(input_text, source_lang)
            except SentimentError as exc:
                # Translation succeeded but sentiment failed -- still show the translation.
                st.warning(str(exc))

        if not translation_failed:
            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.markdown('<div class="nlp-card"><h4>Translated Text</h4>', unsafe_allow_html=True)
                st.write(translated)
                st.markdown("</div>", unsafe_allow_html=True)

            with result_col2:
                st.markdown('<div class="nlp-card"><h4>Sentiment / Tone</h4>', unsafe_allow_html=True)
                if sentiment_result is None:
                    st.caption("Sentiment analysis unavailable for this request.")
                else:
                    badge_class = (
                        "sentiment-positive"
                        if sentiment_result.label == "POSITIVE"
                        else "sentiment-negative"
                    )
                    st.markdown(
                        f'<span class="sentiment-badge {badge_class}">{sentiment_result.label}</span>',
                        unsafe_allow_html=True,
                    )
                    st.metric("Confidence", f"{sentiment_result.confidence * 100:.1f}%")
                    st.progress(sentiment_result.confidence)
                    if sentiment_result.was_translated_for_analysis:
                        st.markdown(
                            '<div class="translation-note">The sentiment model '
                            "(DistilBERT/SST-2) is English-only. Your text was translated "
                            "to English before scoring — this is a translated-text "
                            "sentiment estimate, not native-language sentiment analysis."
                            "</div>",
                            unsafe_allow_html=True,
                        )
                st.markdown("</div>", unsafe_allow_html=True)

            add_history_entry(
                {
                    "time": timestamp(),
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "source_text": input_text,
                    "translated_text": translated,
                    "sentiment": sentiment_result.label if sentiment_result else "N/A",
                    "confidence": f"{sentiment_result.confidence * 100:.1f}%" if sentiment_result else "N/A",
                }
            )

    st.markdown("---")
    st.subheader("Translation History")

    hcol1, hcol2 = st.columns([5, 1])
    with hcol2:
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    if not st.session_state.history:
        st.caption("No translations yet this session.")
    else:
        table_rows = [
            {
                "Time": h["time"],
                "From": h["source_lang"],
                "To": h["target_lang"],
                "Source": truncate(h["source_text"]),
                "Translation": truncate(h["translated_text"]),
                "Sentiment": h["sentiment"],
                "Confidence": h["confidence"],
            }
            for h in st.session_state.history
        ]
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Page: Model Evaluation
# ---------------------------------------------------------------------------
elif page == "Model Evaluation":
    st.markdown('<div class="studio-title">Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="studio-subtitle">BLEU, ROUGE, loss, token-position accuracy, '
        "and perplexity on a fixed evaluation set.</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="nlp-card"><h4>Test Dataset Overview</h4>', unsafe_allow_html=True)
    st.write(
        f"{len(TEST_SAMPLES)} English-source sentence pairs across German, Hindi, "
        "Urdu, and Russian targets."
    )
    overview_rows = [
        {
            "Source Lang": s["source_lang"],
            "Target Lang": s["target_lang"],
            "Source": s["source"],
            "Reference": s["target"],
        }
        for s in TEST_SAMPLES
    ]
    st.dataframe(overview_rows, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.info(
        "**Methodology note:** BLEU and ROUGE are the most meaningful scores here. "
        "Loss and perplexity depend on the exact tokenizer used. The 'accuracy' metric "
        "is a naive token-position match (see README) and should not be read as a "
        "rigorous translation-quality measure — it's included for continuity with the "
        "original project.",
        icon="ℹ️",
    )

    run_clicked = st.button("Run Evaluation", type="primary")

    if run_clicked:
        try:
            with st.spinner("Running evaluation — this loads several translation models..."):
                summary = run_evaluation()
        except Exception as exc:  # noqa: BLE001
            st.error(
                "Evaluation failed while loading models or scoring examples. "
                "Please try again."
            )
        else:
            st.markdown('<div class="nlp-card"><h4>Aggregate Metrics</h4>', unsafe_allow_html=True)
            rouge = summary.avg_rouge
            metric_rows = [
                {"Metric": "BLEU", "Score": f"{summary.avg_bleu:.4f}"},
                {"Metric": "ROUGE-1 (F1)", "Score": f"{rouge['rouge-1']:.4f}"},
                {"Metric": "ROUGE-2 (F1)", "Score": f"{rouge['rouge-2']:.4f}"},
                {"Metric": "ROUGE-L (F1)", "Score": f"{rouge['rouge-l']:.4f}"},
                {"Metric": "Accuracy (token-position, naive)", "Score": f"{summary.avg_accuracy * 100:.2f}%"},
                {"Metric": "Loss", "Score": f"{summary.avg_loss:.4f}"},
                {"Metric": "Perplexity", "Score": f"{summary.avg_perplexity:.4f}"},
            ]
            st.dataframe(metric_rows, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="nlp-card"><h4>Example-Level Results</h4>', unsafe_allow_html=True)
            for ex in summary.examples:
                with st.expander(f"{ex.source_lang} → {ex.target_lang}: {truncate(ex.source, 50)}"):
                    st.write(f"**Source:** {ex.source}")
                    st.write(f"**Reference:** {ex.target}")
                    st.write(f"**Model output:** {ex.model_output}")
                    ex_cols = st.columns(4)
                    ex_cols[0].metric("BLEU", f"{ex.bleu_score:.3f}")
                    ex_cols[1].metric("ROUGE-L", f"{ex.rouge_scores['rouge-l']['f']:.3f}")
                    ex_cols[2].metric("Loss", f"{ex.loss:.3f}")
                    ex_cols[3].metric("Accuracy", f"{ex.accuracy * 100:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
else:
    st.markdown('<div class="studio-title">About This Project</div>', unsafe_allow_html=True)
    st.markdown(
        "Multilingual NLP Studio combines Hugging Face MarianMT translation models "
        "with a DistilBERT sentiment classifier behind a Streamlit interface. "
        "It began as a PyQt5 desktop prototype and was refactored into this modular "
        "web application, separating NLP/business logic (`src/`) from the interface "
        "(`app.py`)."
    )
    st.markdown("See the project README for full architecture, evaluation methodology, "
                "and setup instructions.")
