"""
Basic tests for the core NLP logic.

Per the project requirements, these avoid downloading large models where
possible by mocking the model-loading layer. Tests that would require an
actual MarianMT model are marked and can be run separately/manually.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.translation import (
    ENGLISH,
    InputTooLongError,
    LoadedModel,
    MAX_INPUT_CHARS,
    TranslationError,
    UnsupportedLanguageError,
    translate_text,
)


# ---------------------------------------------------------------------------
# Pure logic tests -- no model needed
# ---------------------------------------------------------------------------

def test_same_language_returns_input_unchanged():
    result = translate_text("Hello there", ENGLISH, ENGLISH)
    assert result == "Hello there"


def test_empty_input_raises():
    with pytest.raises(TranslationError):
        translate_text("", ENGLISH, "German")


def test_whitespace_only_input_raises():
    with pytest.raises(TranslationError):
        translate_text("   \n  ", ENGLISH, "German")


def test_unsupported_language_raises():
    with pytest.raises(UnsupportedLanguageError):
        translate_text("Hello", ENGLISH, "Klingon")


def test_input_too_long_raises():
    huge_text = "a " * (MAX_INPUT_CHARS)  # well over the character limit
    with pytest.raises(InputTooLongError):
        translate_text(huge_text, ENGLISH, "German")


# ---------------------------------------------------------------------------
# Mocked-model tests -- exercise translate_text's routing logic without
# actually downloading MarianMT weights.
# ---------------------------------------------------------------------------

def _mock_loaded_model(decoded_output: str) -> LoadedModel:
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
    mock_model.generate.return_value = [MagicMock()]
    mock_tokenizer.decode.return_value = decoded_output

    return LoadedModel(model=mock_model, tokenizer=mock_tokenizer)


@patch("src.translation.get_model")
def test_english_to_target_uses_direct_model(mock_get_model):
    mock_get_model.return_value = _mock_loaded_model("Hallo Welt")
    result = translate_text("Hello world", ENGLISH, "German")
    assert result == "Hallo Welt"
    mock_get_model.assert_called_once()


@patch("src.translation.get_model")
def test_target_english_uses_direct_model(mock_get_model):
    mock_get_model.return_value = _mock_loaded_model("Hello world")
    result = translate_text("Hallo Welt", "German", ENGLISH)
    assert result == "Hello world"
    mock_get_model.assert_called_once()


@patch("src.translation.get_model")
def test_non_english_to_non_english_pivots_through_english(mock_get_model):
    # First call: German -> English, second call: English -> Hindi
    mock_get_model.side_effect = [
        _mock_loaded_model("Hello world"),
        _mock_loaded_model("नमस्ते दुनिया"),
    ]
    result = translate_text("Hallo Welt", "German", "Hindi")
    assert result == "नमस्ते दुनिया"
    assert mock_get_model.call_count == 2


# ---------------------------------------------------------------------------
# Sentiment output structure (mocked pipeline, no model download)
# ---------------------------------------------------------------------------

@patch("src.sentiment.get_sentiment_pipeline")
@patch("src.sentiment.translate_text")
def test_sentiment_output_structure_for_english(mock_translate, mock_get_pipeline):
    from src.sentiment import analyze_sentiment

    mock_pipeline = MagicMock(return_value=[{"label": "POSITIVE", "score": 0.97}])
    mock_get_pipeline.return_value = mock_pipeline

    result = analyze_sentiment("I love this!", ENGLISH)

    assert result.label == "POSITIVE"
    assert 0.0 <= result.confidence <= 1.0
    assert result.was_translated_for_analysis is False
    mock_translate.assert_not_called()


@patch("src.sentiment.get_sentiment_pipeline")
@patch("src.sentiment.translate_text")
def test_sentiment_translates_non_english_first(mock_translate, mock_get_pipeline):
    from src.sentiment import analyze_sentiment

    mock_translate.return_value = "I love this!"
    mock_pipeline = MagicMock(return_value=[{"label": "POSITIVE", "score": 0.91}])
    mock_get_pipeline.return_value = mock_pipeline

    result = analyze_sentiment("Ich liebe das!", "German")

    assert result.was_translated_for_analysis is True
    assert result.analyzed_text == "I love this!"
    mock_translate.assert_called_once_with("Ich liebe das!", "German", ENGLISH)


# ---------------------------------------------------------------------------
# Evaluation math -- controlled, model-free examples
# ---------------------------------------------------------------------------

def test_compute_bleu_perfect_match():
    from src.evaluation import compute_bleu_score

    score = compute_bleu_score("the cat sat on the mat", "the cat sat on the mat")
    assert score > 0.99


def test_compute_rouge_scores_structure():
    from src.evaluation import compute_rouge_scores

    scores = compute_rouge_scores("the cat sat on the mat", "the cat sat on the mat")
    assert "rouge-1" in scores and "rouge-l" in scores
    assert scores["rouge-1"]["f"] > 0.99


def test_compute_perplexity_from_zero_loss():
    from src.evaluation import compute_perplexity

    assert compute_perplexity(0.0) == pytest.approx(1.0)
