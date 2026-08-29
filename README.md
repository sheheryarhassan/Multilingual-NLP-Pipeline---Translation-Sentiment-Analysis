# 🌐 Multilingual NLP Studio

**Neural machine translation and sentiment analysis for multilingual text,**
built on Hugging Face MarianMT and DistilBERT, with a full model-evaluation
dashboard (BLEU / ROUGE / loss / perplexity).

<!-- If you deploy this, put the live link here: -->
<!-- 🔗 **[Live Demo](https://your-app-url.streamlit.app)** -->

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## Screenshots

<!--
  Add screenshots after your first local run:
  1. Run: streamlit run app.py
  2. Take screenshots of the Translate & Analyze page and the Model
     Evaluation page
  3. Save them as assets/screenshots/translate.png and
     assets/screenshots/evaluation.png
  4. Uncomment the two lines below
-->
<!-- ![Translate & Analyze](assets/screenshots/translate.png) -->
<!-- ![Model Evaluation](assets/screenshots/evaluation.png) -->

## Overview

Multilingual NLP Studio combines Hugging Face **MarianMT** neural machine
translation with a **DistilBERT** sentiment classifier behind a clean,
SaaS-style Streamlit interface. It began as a PyQt5 desktop prototype and
was rebuilt as a modular web application, with NLP/business logic (`src/`)
fully separated from the UI (`app.py`) — the translation and sentiment
modules have zero Streamlit dependency and are independently unit-tested.

Two things this project demonstrates beyond "call a model, show output":
- **Correct multi-hop translation routing** — non-English↔non-English pairs
  are translated through English as a pivot, with the model direction
  resolved explicitly rather than by convention/coincidence.
- **Honest evaluation methodology** — the evaluation dashboard reports BLEU,
  ROUGE, loss, and perplexity against a fixed test set, and is transparent
  about the limitations of each metric rather than presenting a single
  inflated "accuracy" number (see [Evaluation](#evaluation) below).

It supports translation between English and Urdu, Hindi, German, and
Russian (plus Arabic model mappings, currently untested in the evaluation
set — see [Supported Languages](#supported-languages)), sentiment/tone
analysis of translated or source text, a translation evaluation dashboard
(BLEU, ROUGE, loss, accuracy, perplexity), and a session-based translation
history.

## Features

- Multilingual neural translation via Helsinki-NLP MarianMT models
- Automatic pivot-through-English translation for non-English↔non-English pairs
- Sentiment/tone analysis via DistilBERT (SST-2), with translation-to-English
  handled transparently for non-English input
- Model evaluation dashboard: BLEU, ROUGE-1/2/L, loss, token-position
  accuracy, perplexity, and per-example breakdowns
- Session-based translation history with a clear-history control
- Graceful error handling for empty input, unsupported languages, model
  load failures, and overly long input — no raw tracebacks shown to users

## Architecture

```
User
  ↓
Streamlit (app.py)
  ↓
Translation / Sentiment / Evaluation services (src/)
  ↓
Hugging Face Models (MarianMT, DistilBERT)
  ↓
Results
```

`app.py` contains no NLP logic — it only calls into `src/` and renders
results. `src/translation.py` and `src/sentiment.py` have no Streamlit
dependency, so they can be imported and unit-tested (or reused in a CLI
script) independently of the web interface.

## Technology Stack

- Python
- Streamlit
- Hugging Face Transformers
- MarianMT
- DistilBERT
- PyTorch
- NLTK (BLEU)
- rouge (ROUGE)

## Supported Languages

Translation pairs currently supported (English is the pivot language):

| Language | To-English model | From-English model |
|---|---|---|
| Urdu | Helsinki-NLP/opus-mt-ur-en | Helsinki-NLP/opus-mt-en-ur |
| Hindi | Helsinki-NLP/opus-mt-hi-en | Helsinki-NLP/opus-mt-en-hi |
| German | Helsinki-NLP/opus-mt-de-en | Helsinki-NLP/opus-mt-en-de |
| Russian | Helsinki-NLP/opus-mt-ru-en | Helsinki-NLP/opus-mt-en-ru |
| Arabic | Helsinki-NLP/opus-mt-ar-en | Helsinki-NLP/opus-mt-en-ar |

Arabic model mappings are carried over from the original project but are
not currently included in the fixed evaluation set in `src/evaluation.py`.

## Translation Pipeline

- **Direct pairs** (English ↔ any supported language) use a single MarianMT
  model in the appropriate direction.
- **Non-English → non-English** pairs (e.g. German → Hindi) are translated
  in two hops: source → English → target, since no direct MarianMT model
  exists for most such pairs. This mirrors the original desktop app's
  behavior.
- Models and tokenizers are loaded lazily on first use and cached for the
  life of the server process (see [Caching Strategy](#caching-strategy)).

## Sentiment Pipeline

The sentiment model, `distilbert-base-uncased-finetuned-sst-2-english`, was
trained only on English text (the SST-2 movie review dataset). It does
**not** perform native multilingual sentiment analysis.

For non-English input, the app translates the text to English first (using
the same translation pipeline above) and then runs sentiment analysis on
the English result. The UI labels this explicitly whenever it happens, so
results are never presented as native-language sentiment.

## Evaluation

The Model Evaluation page runs a fixed set of English-source sentence pairs
through the translation pipeline and reports:

- **BLEU** — n-gram precision against the reference translation
- **ROUGE-1 / ROUGE-2 / ROUGE-L** — recall-oriented n-gram/longest-common-
  subsequence overlap
- **Loss** — cross-entropy loss of the MarianMT model predicting the
  reference translation given the source
- **Perplexity** — `exp(loss)`
- **Accuracy** — a **naive token-position match**: both the model output
  and the reference are tokenized, truncated to the shorter of the two
  sequence lengths, and compared position-by-position. This metric is
  included only for continuity with the original project's methodology —
  it is highly sensitive to any length or word-order mismatch and should
  **not** be interpreted as a rigorous translation-quality score. BLEU and
  ROUGE are more meaningful here.

### Evaluation Methodology Fixes

The original `Testing.py` had three issues that are corrected in
`src/evaluation.py`:

1. **Missing imports.** `Testing.py` referenced `language_models`,
   `translate_text`, `MarianMTModel`, and `MarianTokenizer` without
   importing any of them — running it as-is raises `NameError`.
2. **Wrong model direction for evaluation.** The original selected
   `language_models[source_lang][1]` for both the model and tokenizer name.
   Index `[1]` is the *English→lang* model; this only "worked" because
   every test sample happened to have `source_lang == "English"`. The
   fixed version resolves the model direction explicitly from
   `source_lang`/`target_lang`, the same way the translation module does.
3. **Label truncation bug in loss computation.** The original truncated the
   *target* label sequence to the *source* input's token length
   (`labels[:, :inputs["input_ids"].shape[1]]`). Two sentences in two
   different languages have no reason to tokenize to the same length, so
   this silently discarded part of the reference translation whenever the
   target was longer than the source, corrupting the loss value. The fixed
   version tokenizes labels to their own natural length and only masks
   padding tokens with `-100`.

## Project Structure

```
multilingual-nlp-studio/
│
├── app.py                     # Streamlit UI only — no NLP logic
│
├── src/
│   ├── __init__.py
│   ├── translation.py         # MarianMT translation logic
│   ├── sentiment.py           # DistilBERT sentiment logic
│   ├── evaluation.py          # BLEU/ROUGE/loss/accuracy/perplexity
│   └── utils.py                # small shared helpers
│
├── tests/
│   └── test_translation.py    # unit tests (model-loading is mocked)
│
├── assets/
│   └── screenshots/            # add screenshots here for the README
│
├── notebooks/                  # optional exploration notebooks
│
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

## Installation

```bash
git clone https://github.com/<your-username>/multilingual-nlp-studio.git
cd multilingual-nlp-studio

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Models download from Hugging
Face on first use (this can take a few minutes depending on your
connection) and are then cached in memory for the rest of the session.

## Run Tests

```bash
pytest tests/
```

Tests that exercise `translate_text`'s routing logic mock the model-loading
layer, so the suite runs quickly without downloading MarianMT weights.

## Caching Strategy

`src/translation.py` and `src/sentiment.py` keep their own module-level
dict caches (`_model_cache`, `_sentiment_pipeline`). Streamlit re-runs the
script body on every user interaction but does **not** reload Python
modules within the same server process, so this module-level cache already
persists across reruns for free: a model downloaded on the first
translation stays resident in memory for the life of the process and is
reused by every later request. This keeps the NLP modules Streamlit-agnostic
(they can be imported and unit-tested with zero Streamlit dependency) while
still getting "load once, reuse everywhere" behavior in the running app.
No model is loaded until it's actually needed by a translation or sentiment
request — nothing is eagerly loaded at startup.

## Future Improvements

- Add automated language detection instead of requiring manual source-language selection
- Persist translation history to a lightweight database for cross-session history
- Add batch translation (file upload) support
- Expand the evaluation set to include Arabic and non-English source pairs
- Add a confidence interval or bootstrap estimate around evaluation metrics

## Limitations

- Sentiment analysis is English-only; non-English results depend on
  translation quality first.
- MarianMT and DistilBERT model downloads can take significant time and
  memory on first run, which may affect cold-start latency on constrained
  deployment tiers.
- The evaluation set is small (12 fixed sentence pairs) and is meant to
  demonstrate the evaluation pipeline, not to serve as a statistically
  rigorous benchmark.

## Author

Sheheryar Hassan
M.S. Data Sciences and Statistics
The University of Texas at Dallas
