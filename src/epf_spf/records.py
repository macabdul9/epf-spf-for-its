from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MCQRecord:
    """One multiple-choice question with its audio.

    `answer_index` is None when the gold text matches no option; such items are
    ungradeable and dropped by the loaders.
    """

    unique_id: str
    question: str
    choices: list[str]
    answer: str
    audio_paths: list[str]
    category: str
    length_type: str
    answer_index: int | None
