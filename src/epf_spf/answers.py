"""Map model output and gold answers to option indices.

Behaviour is identical to the previous implementation
(`entropic-particle-filter/benchmarking/mmau_pro/scoring.py`); tests/test_parity.py
checks that.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

LETTERS = "ABCDEFGHIJK"


def normalize(text: str | None) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace."""
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def match_answer_index(answer: str, choices: list[str], threshold: float = 0.85) -> int | None:
    """Index of the gold `answer` text in `choices`: exact after normalisation, else the
    best fuzzy match at or above `threshold`, else None (ungradeable)."""
    if not choices:
        return None
    normalized_answer = normalize(answer)
    normalized_choices = [normalize(c) for c in choices]
    if normalized_answer in normalized_choices:
        return normalized_choices.index(normalized_answer)
    best_index, best_ratio = None, 0.0
    for i, choice in enumerate(normalized_choices):
        ratio = SequenceMatcher(None, normalized_answer, choice).ratio()
        if ratio > best_ratio:
            best_index, best_ratio = i, ratio
    return best_index if best_ratio >= threshold else None


def _is_option_token(match: re.Match, text: str) -> bool:
    # "A" and "I" are also English words ("A piano", "I believe"); accept them as
    # option letters only when followed by punctuation or the end of a line.
    letter = match.group(1).upper()
    if letter not in ("A", "I"):
        return True
    rest = text[match.end() :]
    return re.match(r"\s*([).,:;!?\]]|$|\n)", rest) is not None


def extract_letter(text: str, num_choices: int) -> int | None:
    """Chosen option index from model output, or None.

    Precedence: the last `\\boxed{X}`, then the last `Answer: X`, then the last
    standalone uppercase option letter.
    """
    if not text:
        return None
    valid = LETTERS[:num_choices]
    for match in reversed(list(re.finditer(r"\\boxed\s*\{\s*\(?([A-K])", text, re.IGNORECASE))):
        letter = match.group(1).upper()
        if letter in valid:
            return valid.index(letter)
    for match in reversed(
        list(re.finditer(r"answer\s*[:\-]?\s*\(?([A-K])\b", text, re.IGNORECASE))
    ):
        letter = match.group(1).upper()
        if letter in valid and _is_option_token(match, text):
            return valid.index(letter)
    for match in reversed(list(re.finditer(r"\b([A-K])\b", text))):
        letter = match.group(1)
        if letter in valid and _is_option_token(match, text):
            return valid.index(letter)
    return None


def predicted_index(text: str, choices: list[str], text_threshold: float = 0.6) -> int | None:
    """The model's chosen option: by letter, else the longest option text contained in
    the output, else the best fuzzy match at or above `text_threshold`."""
    index = extract_letter(text, len(choices))
    if index is not None:
        return index
    normalized_text = normalize(text)
    normalized_choices = [(i, normalize(c)) for i, c in enumerate(choices)]
    # Prefer the longest contained option so "dog" cannot shadow "dog barking".
    contained = [(i, c) for i, c in normalized_choices if c and c in normalized_text]
    if contained:
        return max(contained, key=lambda pair: len(pair[1]))[0]
    best_index, best_ratio = None, 0.0
    for i, choice in normalized_choices:
        ratio = SequenceMatcher(None, choice, normalized_text).ratio()
        if ratio > best_ratio:
            best_index, best_ratio = i, ratio
    return best_index if best_ratio >= text_threshold else None
