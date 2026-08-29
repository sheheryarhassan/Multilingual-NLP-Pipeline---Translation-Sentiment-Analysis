"""
evaluation.py
-------------
Translation evaluation metrics, refactored from the original Testing.py.

Fixes applied relative to the original script (see README "Evaluation
Methodology" section for the full explanation):

1. Testing.py referenced `language_models`, `translate_text`, `MarianMTModel`,
   and `MarianTokenizer` without importing any of them -- it would have
   raised NameError immediately. Everything needed is now imported from
   src.translation.

2. Testing.py selected the model/tokenizer for loss computation using
   `language_models[source_lang][1]` for BOTH the model and tokenizer name.
   Index [1] is the English->lang direction. In the original test set
   source_lang is always "English", so this happened to select a real
   model, but for the wrong reason -- it would silently break the moment a
   non-English source was tested. This module resolves the model direction
   explicitly based on source_lang/target_lang, using the same logic as
   the translation module.

3. `compute_translation_loss` truncated the TARGET label sequence to the
   length of the SOURCE input sequence:
       labels = labels[:, :inputs["input_ids"].shape[1]]
   Two sentences in two different languages have no reason to tokenize to
   the same length, so this discarded part of the reference translation
   for any target sentence longer than the source, corrupting the loss
   value. Labels are now tokenized to their own natural length; only
   padding tokens are masked to -100.

4. `compute_accuracy` is preserved as a metric (per instructions not to
   silently change methodology) but is clearly documented and labeled as
   a naive token-position match: it truncates both sequences to the
   shorter length and checks position-by-position equality. This is very
   sensitive to any length or alignment mismatch and should NOT be read as
   a rigorous translation-quality metric. It is reported here for
   continuity with the original project, not because it is scientifically
   strong -- BLEU and ROUGE are the more meaningful scores in this
   dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

from src.translation import ENGLISH, LANGUAGE_MODELS, get_model, translate_text

# ---------------------------------------------------------------------------
# Fixed evaluation set, preserved from the original Testing.py.
# ---------------------------------------------------------------------------
TEST_SAMPLES: List[Dict[str, str]] = [
    {"source": "Hello, how are you?", "source_lang": "English", "target_lang": "German", "target": "Hallo, wie geht es dir?"},
    {"source": "I am going to the market.", "source_lang": "English", "target_lang": "German", "target": "Ich gehe zum Markt."},
    {"source": "What is your name?", "source_lang": "English", "target_lang": "German", "target": "Wie heißt du?"},
    {"source": "The weather is nice today.", "source_lang": "English", "target_lang": "Hindi", "target": "आज का मौसम अच्छा है।"},
    {"source": "Do you like playing cricket?", "source_lang": "English", "target_lang": "Hindi", "target": "क्या आपको क्रिकेट खेलना पसंद है?"},
    {"source": "We are learning machine translation.", "source_lang": "English", "target_lang": "Hindi", "target": "हम मशीन अनुवाद सीख रहे हैं।"},
    {"source": "Good morning! How are you?", "source_lang": "English", "target_lang": "Urdu", "target": "صبح بخیر! آپ کیسے ہیں؟"},
    {"source": "This book is very interesting.", "source_lang": "English", "target_lang": "Urdu", "target": "یہ کتاب بہت دلچسپ ہے۔"},
    {"source": "Please help me with this task.", "source_lang": "English", "target_lang": "Urdu", "target": "براہ کرم اس کام میں میری مدد کریں۔"},
    {"source": "Can you speak Russian?", "source_lang": "English", "target_lang": "Russian", "target": "Вы говорите по-русски?"},
    {"source": "I enjoy reading books.", "source_lang": "English", "target_lang": "Russian", "target": "Мне нравится читать книги."},
    {"source": "She is a very talented musician.", "source_lang": "English", "target_lang": "Russian", "target": "Она очень талантливая музыкантка."},
]


@dataclass
class ExampleResult:
    source: str
    target: str
    source_lang: str
    target_lang: str
    model_output: str
    loss: float
    bleu_score: float
    rouge_scores: dict
    accuracy: float
    perplexity: float


@dataclass
class EvaluationSummary:
    examples: List[ExampleResult] = field(default_factory=list)

    @property
    def avg_bleu(self) -> float:
        return float(np.mean([e.bleu_score for e in self.examples])) if self.examples else 0.0

    @property
    def avg_loss(self) -> float:
        return float(np.mean([e.loss for e in self.examples])) if self.examples else 0.0

    @property
    def avg_accuracy(self) -> float:
        return float(np.mean([e.accuracy for e in self.examples])) if self.examples else 0.0

    @property
    def avg_perplexity(self) -> float:
        return float(np.mean([e.perplexity for e in self.examples])) if self.examples else 0.0

    @property
    def avg_rouge(self) -> Dict[str, float]:
        if not self.examples:
            return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
        out = {}
        for key in ("rouge-1", "rouge-2", "rouge-l"):
            out[key] = float(np.mean([e.rouge_scores[key]["f"] for e in self.examples]))
        return out


def _resolve_eval_model_name(source_lang: str, target_lang: str) -> str:
    """
    Pick the correct HF model name for computing loss/perplexity on a
    source->target pair. Mirrors src.translation's direction logic instead
    of the original script's incorrect fixed index [1].
    """
    if target_lang == ENGLISH:
        return LANGUAGE_MODELS[source_lang][0]
    if source_lang == ENGLISH:
        return LANGUAGE_MODELS[target_lang][1]
    raise ValueError("Evaluation set only contains English-source pairs currently.")


def compute_translation_loss(model, tokenizer, source_text: str, target_text: str) -> float:
    """
    Compute cross-entropy loss of `model` predicting `target_text` from
    `source_text`.

    Fix vs. original: labels are tokenized to their own natural length
    (not truncated to the source's token length). Only pad tokens are
    masked with -100, which is the standard way to tell the loss function
    to ignore padding without discarding real target tokens.
    """
    inputs = tokenizer(source_text, return_tensors="pt", padding=True, truncation=True)
    labels = tokenizer(target_text, return_tensors="pt", padding=True, truncation=True).input_ids
    labels = labels.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    outputs = model(**inputs, labels=labels)
    return float(outputs.loss.item())


def compute_bleu_score(reference: str, hypothesis: str) -> float:
    reference_tokens = [reference.split()]
    hypothesis_tokens = hypothesis.split()
    return sentence_bleu(
        reference_tokens, hypothesis_tokens, smoothing_function=SmoothingFunction().method1
    )


def compute_rouge_scores(reference: str, hypothesis: str) -> dict:
    rouge = Rouge()
    # rouge-score library expects (hypothesis, reference) in that order.
    return rouge.get_scores(hypothesis, reference, avg=True)


def compute_accuracy(model_output: str, ground_truth: str, tokenizer) -> float:
    """
    Naive token-position match accuracy (see module docstring for caveats).
    Kept for continuity with the original project's methodology.
    """
    model_tokens = tokenizer(model_output, return_tensors="pt", padding=True).input_ids
    ground_truth_tokens = tokenizer(ground_truth, return_tensors="pt", padding=True).input_ids
    length = min(model_tokens.shape[1], ground_truth_tokens.shape[1])
    if length == 0:
        return 0.0
    model_tokens = model_tokens[:, :length]
    ground_truth_tokens = ground_truth_tokens[:, :length]
    correct = (model_tokens == ground_truth_tokens).sum().item()
    total = ground_truth_tokens.numel()
    return correct / total if total else 0.0


def compute_perplexity(loss: float) -> float:
    return float(np.exp(loss))


def evaluate_sample(sample: Dict[str, str]) -> ExampleResult:
    """Run the full evaluation pipeline on a single test sample."""
    source_lang = sample["source_lang"]
    target_lang = sample["target_lang"]
    source_text = sample["source"]
    target_text = sample["target"]

    model_name = _resolve_eval_model_name(source_lang, target_lang)
    loaded = get_model(model_name)

    model_output = translate_text(source_text, source_lang, target_lang)
    loss = compute_translation_loss(loaded.model, loaded.tokenizer, source_text, target_text)
    bleu = compute_bleu_score(target_text, model_output)
    rouge_scores = compute_rouge_scores(target_text, model_output)
    accuracy = compute_accuracy(model_output, target_text, loaded.tokenizer)
    perplexity = compute_perplexity(loss)

    return ExampleResult(
        source=source_text,
        target=target_text,
        source_lang=source_lang,
        target_lang=target_lang,
        model_output=model_output,
        loss=loss,
        bleu_score=bleu,
        rouge_scores=rouge_scores,
        accuracy=accuracy,
        perplexity=perplexity,
    )


def run_evaluation(samples: List[Dict[str, str]] = None) -> EvaluationSummary:
    """Run evaluation over all (or a subset of) TEST_SAMPLES."""
    samples = samples if samples is not None else TEST_SAMPLES
    summary = EvaluationSummary()
    for sample in samples:
        summary.examples.append(evaluate_sample(sample))
    return summary
