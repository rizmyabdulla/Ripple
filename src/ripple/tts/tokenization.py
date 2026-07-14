"""Multilingual tokenization with explicit grapheme and UTF-8 fallbacks."""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from .normalization import MultilingualTextNormalizer

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
BYTE_OFFSET = 4
GRAPHEME_OFFSET = 260


@dataclass(frozen=True)
class TextTokens:
    ids: tuple[int, ...]
    pieces: tuple[str, ...]
    language: str
    mode: str


def graphemes(text: str) -> tuple[str, ...]:
    """Small Unicode grapheme approximation with combining/ZWJ support."""

    clusters: list[str] = []
    join_next = False
    for char in text:
        combining = bool(unicodedata.combining(char))
        variation = "\ufe00" <= char <= "\ufe0f"
        if clusters and (combining or variation or join_next or char == "\u200d"):
            clusters[-1] += char
        else:
            clusters.append(char)
        join_next = char == "\u200d"
    return tuple(clusters)


class MultilingualTokenizer:
    """Tokenize with optional local phonemizers and deterministic fallbacks.

    Unknown graphemes are represented as UTF-8 bytes, making every valid
    Unicode string encodable without changing the vocabulary.
    """

    def __init__(
        self,
        vocabulary: Mapping[str, int] | None = None,
        *,
        phonemizers: Mapping[str, Callable[[str], Iterable[str]]] | None = None,
        normalizer: MultilingualTextNormalizer | None = None,
    ) -> None:
        self.vocabulary = dict(vocabulary or {})
        if any(token_id < GRAPHEME_OFFSET for token_id in self.vocabulary.values()):
            raise ValueError(f"custom token ids must be >= {GRAPHEME_OFFSET}")
        self.phonemizers = dict(phonemizers or {})
        self.normalizer = normalizer or MultilingualTextNormalizer()

    def encode(
        self,
        text: str,
        *,
        language: str | None = None,
        use_phonemes: bool = True,
        add_special_tokens: bool = True,
    ) -> TextTokens:
        normalized = self.normalizer.normalize(text, language)
        source_pieces: tuple[str, ...]
        mode = "grapheme-byte"
        phonemizer = self.phonemizers.get(normalized.language)
        if use_phonemes and phonemizer is not None:
            source_pieces = tuple(phonemizer(normalized.text))
            mode = "phoneme"
        else:
            source_pieces = graphemes(normalized.text)

        ids: list[int] = [BOS_ID] if add_special_tokens else []
        pieces: list[str] = ["<bos>"] if add_special_tokens else []
        for piece in source_pieces:
            token_id = self.vocabulary.get(piece)
            if token_id is not None:
                ids.append(token_id)
                pieces.append(piece)
                continue
            for byte in piece.encode("utf-8"):
                ids.append(BYTE_OFFSET + byte)
                pieces.append(f"<0x{byte:02x}>")
        if add_special_tokens:
            ids.append(EOS_ID)
            pieces.append("<eos>")
        return TextTokens(tuple(ids), tuple(pieces), normalized.language, mode)
