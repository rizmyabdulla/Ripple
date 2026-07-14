"""Text-based intelligibility metrics; ASR transcription remains pluggable."""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, Sequence


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_item in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str, normalizer: Callable[[str], str] = normalize_text) -> float:
    reference_words = normalizer(reference).split()
    hypothesis_words = normalizer(hypothesis).split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return edit_distance(reference_words, hypothesis_words) / len(reference_words)


def character_error_rate(reference: str, hypothesis: str, normalizer: Callable[[str], str] = normalize_text) -> float:
    reference_chars = list(normalizer(reference).replace(" ", ""))
    hypothesis_chars = list(normalizer(hypothesis).replace(" ", ""))
    if not reference_chars:
        return 0.0 if not hypothesis_chars else 1.0
    return edit_distance(reference_chars, hypothesis_chars) / len(reference_chars)


def corpus_error_rate(pairs: Sequence[tuple[str, str]], *, level: str = "word") -> float:
    total_errors = 0
    total_units = 0
    for reference, hypothesis in pairs:
        normalized_reference = normalize_text(reference)
        normalized_hypothesis = normalize_text(hypothesis)
        if level == "word":
            reference_units = normalized_reference.split()
            hypothesis_units = normalized_hypothesis.split()
        elif level == "character":
            reference_units = list(normalized_reference.replace(" ", ""))
            hypothesis_units = list(normalized_hypothesis.replace(" ", ""))
        else:
            raise ValueError("level must be word or character")
        total_errors += edit_distance(reference_units, hypothesis_units)
        total_units += len(reference_units)
    return total_errors / total_units if total_units else float(total_errors > 0)
