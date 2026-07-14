"""Deterministic, dependency-free multilingual text normalization."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

_SPACE_RE = re.compile(r"\s+")
_CONTROL_CATEGORIES = {"Cc", "Cf", "Cs", "Co", "Cn"}
_BOUNDARY_PUNCTUATION = frozenset(".!?\u3002\uff01\uff1f\u061f")


@dataclass(frozen=True)
class NormalizedText:
    """Normalized text and the language hint used to process it."""

    text: str
    language: str


class MultilingualTextNormalizer:
    """Normalize Unicode while retaining script, accents, and punctuation.

    The normalizer intentionally does not spell out numbers or abbreviations:
    those operations are language-specific and silently applying English rules
    to unknown text is worse than preserving the original graphemes.
    """

    def __init__(
        self,
        *,
        form: Literal["NFC", "NFKC"] = "NFKC",
        lowercase: bool = False,
    ) -> None:
        if form not in {"NFC", "NFKC"}:
            raise ValueError("form must be NFC or NFKC")
        self.form = form
        self.lowercase = lowercase

    def normalize(self, text: str, language: str | None = None) -> NormalizedText:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        normalized = unicodedata.normalize(self.form, text)
        normalized = "".join(
            " " if char in "\r\n\t" else char
            for char in normalized
            if char in "\r\n\t" or unicodedata.category(char) not in _CONTROL_CATEGORIES
        )
        normalized = _SPACE_RE.sub(" ", normalized).strip()
        if self.lowercase:
            normalized = normalized.casefold()
        return NormalizedText(normalized, canonical_language(language))


def canonical_language(language: str | None) -> str:
    """Return a conservative BCP-47-like language tag."""

    if language is None or not language.strip():
        return "und"
    parts = language.strip().replace("_", "-").split("-")
    if not all(part.isalnum() for part in parts):
        raise ValueError(f"invalid language tag: {language!r}")
    return "-".join([parts[0].lower(), *(part.lower() for part in parts[1:])])


def is_safe_boundary(text: str) -> bool:
    """Whether incremental text ends at a stable planning boundary."""

    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in _BOUNDARY_PUNCTUATION
