"""
sentiment.py
------------
Sentiment/tone analysis using distilbert-base-uncased-finetuned-sst-2-english.

Important: this model is English-only. It was trained on English movie
review data (SST-2) and produces meaningless or misleading results on other
languages. The original desktop app ran sentiment analysis on already
English-translated text for non-English->non-English pairs, but for
English->non-English or same-language cases it was less clear what text was
actually being scored. This module makes the rule explicit and consistent:

    Sentiment is ALWAYS computed on the English-language form of the input,
    obtained via translation.translate_text() if the source language is not
    already English.

The caller (app.py) is responsible for telling the UI honestly that a
translation-then-sentiment step happened for non-English input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from transformers import pipeline

from src.translation import ENGLISH, translate_text, TranslationError

SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

_sentiment_pipeline = None  # lazily initialized, see get_sentiment_pipeline()


class SentimentError(Exception):
    """Raised when sentiment analysis fails for a reason the UI should show."""


@dataclass
class SentimentResult:
    label: str  # "POSITIVE" or "NEGATIVE"
    confidence: float  # 0.0 - 1.0, the model's confidence in `label`
    analyzed_text: str  # the (possibly translated) English text that was scored
    was_translated_for_analysis: bool  # True if input wasn't already English


def get_sentiment_pipeline():
    """Lazily construct and cache the HF sentiment-analysis pipeline."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        try:
            _sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=SENTIMENT_MODEL_NAME,
                revision="main",
            )
        except Exception as exc:  # noqa: BLE001
            raise SentimentError(
                "Unable to load the sentiment analysis model. Please try again."
            ) from exc
    return _sentiment_pipeline


def analyze_sentiment(text: str, source_lang: str) -> SentimentResult:
    """
    Analyze the sentiment of `text`, translating to English first if needed.

    Raises:
        SentimentError, TranslationError
    """
    if not text or not text.strip():
        raise SentimentError("Please enter text before analyzing sentiment.")

    was_translated = source_lang != ENGLISH
    text_for_analysis = text
    if was_translated:
        text_for_analysis = translate_text(text, source_lang, ENGLISH)

    nlp = get_sentiment_pipeline()
    try:
        result = nlp(text_for_analysis[:512])[0]  # DistilBERT context limit guard
    except Exception as exc:  # noqa: BLE001
        raise SentimentError(
            "Sentiment analysis failed while processing your text. Please try again."
        ) from exc

    label = result["label"].upper()
    confidence = float(result["score"])

    return SentimentResult(
        label=label,
        confidence=confidence,
        analyzed_text=text_for_analysis,
        was_translated_for_analysis=was_translated,
    )
