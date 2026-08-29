"""
translation.py
---------------
Neural machine translation logic built on Hugging Face MarianMT models.

This module is UI-agnostic: it has no dependency on Streamlit, PyQt5, or any
other interface. It can be imported by the web app, the evaluation module,
or the test suite equally.

Design notes / what changed vs. the original translator_app.py:
- The original ``language_models`` dict had a bogus ``"English"`` entry that
  duplicated German model names and was never actually used by the routing
  logic. It has been removed. English is now treated explicitly as the
  pivot language in ``translate_text``.
- Model/tokenizer objects are loaded lazily and cached in a module-level
  dict, same pattern as the original app. Streamlit's ``st.cache_resource``
  is layered on top of this in app.py so models persist across reruns/users
  within a session, not just within a single script execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from transformers import MarianMTModel, MarianTokenizer

# ---------------------------------------------------------------------------
# Supported languages and their MarianMT model identifiers.
#
# Convention: language_models[lang] = (model_to_english, model_from_english)
#   - model_to_english:   translates FROM this language INTO English
#   - model_from_english: translates FROM English INTO this language
#
# English itself is intentionally NOT a key here -- it is the pivot
# language, not something you translate "into English" or "out of English
# into English". Treating it as a special case in translate_text() avoids
# the dead/misleading entry that existed in the original code.
# ---------------------------------------------------------------------------
LANGUAGE_MODELS: Dict[str, Tuple[str, str]] = {
    "Urdu": ("Helsinki-NLP/opus-mt-ur-en", "Helsinki-NLP/opus-mt-en-ur"),
    "Hindi": ("Helsinki-NLP/opus-mt-hi-en", "Helsinki-NLP/opus-mt-en-hi"),
    "German": ("Helsinki-NLP/opus-mt-de-en", "Helsinki-NLP/opus-mt-en-de"),
    "Russian": ("Helsinki-NLP/opus-mt-ru-en", "Helsinki-NLP/opus-mt-en-ru"),
    "Arabic": ("Helsinki-NLP/opus-mt-ar-en", "Helsinki-NLP/opus-mt-en-ar"),
}

ENGLISH = "English"
SUPPORTED_LANGUAGES = [ENGLISH] + sorted(LANGUAGE_MODELS.keys())

MAX_INPUT_CHARS = 2000  # guardrail against pathologically long input


class TranslationError(Exception):
    """Raised for any translation failure that the UI should show cleanly."""


class UnsupportedLanguageError(TranslationError):
    """Raised when a requested language is not in SUPPORTED_LANGUAGES."""


class ModelLoadError(TranslationError):
    """Raised when a MarianMT model/tokenizer fails to download or load."""


class InputTooLongError(TranslationError):
    """Raised when input text exceeds MAX_INPUT_CHARS."""


@dataclass
class LoadedModel:
    model: MarianMTModel
    tokenizer: MarianTokenizer


# Module-level cache: {huggingface_model_name: LoadedModel}
# Streamlit's own caching wraps the public functions below, but this cache
# is kept too so the module works correctly even when used outside Streamlit
# (e.g. in the evaluation module or in tests).
_model_cache: Dict[str, LoadedModel] = {}


def get_model(model_name: str) -> LoadedModel:
    """Lazily load and cache a MarianMT model + tokenizer by HF model name."""
    if model_name not in _model_cache:
        try:
            model = MarianMTModel.from_pretrained(model_name)
            tokenizer = MarianTokenizer.from_pretrained(model_name)
        except Exception as exc:  # noqa: BLE001 - we deliberately catch broadly
            raise ModelLoadError(
                f"Could not load translation model '{model_name}'. "
                f"Check your internet connection or try again later."
            ) from exc
        _model_cache[model_name] = LoadedModel(model=model, tokenizer=tokenizer)
    return _model_cache[model_name]


def _resolve_model_name(source_lang: str, target_lang: str) -> str:
    """Return the single HF model name that directly handles this pair."""
    if target_lang == ENGLISH:
        return LANGUAGE_MODELS[source_lang][0]
    if source_lang == ENGLISH:
        return LANGUAGE_MODELS[target_lang][1]
    raise ValueError(
        "_resolve_model_name only handles pairs where one side is English; "
        "use translate_text() for the general case."
    )


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate `text` from `source_lang` to `target_lang`.

    Behavior preserved from the original desktop app:
      - identical source/target language returns the text unchanged
      - non-English -> non-English pairs are routed through English as a
        pivot, since no direct MarianMT model exists for most such pairs
      - unsupported languages raise UnsupportedLanguageError instead of
        returning a plain string (the original returned the string
        "Language not supported.", which is fragile because it can't be
        distinguished from a legitimate translation result)

    Raises:
        UnsupportedLanguageError, InputTooLongError, ModelLoadError,
        TranslationError
    """
    if not text or not text.strip():
        raise TranslationError("Please enter text before translating.")

    if len(text) > MAX_INPUT_CHARS:
        raise InputTooLongError(
            f"Input is too long ({len(text)} characters). "
            f"Please shorten it to {MAX_INPUT_CHARS} characters or fewer."
        )

    if source_lang not in SUPPORTED_LANGUAGES or target_lang not in SUPPORTED_LANGUAGES:
        raise UnsupportedLanguageError(
            f"'{source_lang}' or '{target_lang}' is not a supported language."
        )

    if source_lang == target_lang:
        return text

    # Non-English -> non-English: pivot through English.
    if source_lang != ENGLISH and target_lang != ENGLISH:
        intermediate = translate_text(text, source_lang, ENGLISH)
        return translate_text(intermediate, ENGLISH, target_lang)

    model_name = _resolve_model_name(source_lang, target_lang)
    loaded = get_model(model_name)

    try:
        inputs = loaded.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = loaded.model.generate(**inputs, max_new_tokens=512)
        translated_text = loaded.tokenizer.decode(
            translated_tokens[0], skip_special_tokens=True
        )
    except Exception as exc:  # noqa: BLE001
        raise TranslationError(
            "Translation failed while processing your text. Please try again "
            "or try a shorter input."
        ) from exc

    return translated_text
