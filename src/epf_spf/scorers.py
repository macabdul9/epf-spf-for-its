"""Where a particle's score s comes from at each step.

JudgeScorer  an external zero-shot judge (an audio LM) answers Yes/No about the latest
             step; s = P(Yes) renormalised over the Yes/No mass of its first token.
ProbeScorer  the generator itself is interrupted with "Answer:" and its forced choice
             is read off; s = the probability mass on its most likely option letter.
RandomScorer a control: s ~ Uniform(0, 1), independent of the text.

Both follow the previous implementation exactly (`judge_client.py` and
`AnswerProbeEPF._probe` in `probe_epf.py`).
"""

from __future__ import annotations

import asyncio
import math
import zlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from openai import APIError, AsyncOpenAI

from epf_spf import prompts
from epf_spf.answers import LETTERS
from epf_spf.llm import GeneratorClient
from epf_spf.records import MCQRecord
from epf_spf.steps import FINAL_MARKER, StepFormat

JUDGE_TOP_LOGPROBS = 20
_PROBE_TOKEN_PUNCTUATION = ")>].,:;!?*'\"( "


@dataclass
class StepScore:
    """`score` is None when there is no usable signal. `failed` marks a judge call
    that errored after retries (as opposed to a judge that answered neither Yes nor No,
    which is a valid score of 0.5)."""

    score: float | None
    letter_masses: dict[str, float] | None = None
    failed: bool = False


def yes_probability(top_logprobs: list[tuple[str, float]]) -> float:
    """Yes mass over Yes+No mass among the top tokens; 0.5 when neither appears."""
    yes_mass = no_mass = 0.0
    for token, logprob in top_logprobs:
        word = token.strip().strip('."“”').lower()
        if word == "yes":
            yes_mass += math.exp(logprob)
        elif word == "no":
            no_mass += math.exp(logprob)
    total = yes_mass + no_mass
    return 0.5 if total == 0 else yes_mass / total


def letter_masses(positions: list[list[tuple[str, float]]], valid: str) -> dict[str, float] | None:
    """Option-letter probability mass at the first generated position that has any,
    summed over surface variants ("B", " B", "B)")."""
    for top in positions:
        masses: dict[str, float] = defaultdict(float)
        for token, logprob in top:
            stripped = token.strip(_PROBE_TOKEN_PUNCTUATION)
            if len(stripped) == 1 and stripped.upper() in valid:
                masses[stripped.upper()] += math.exp(logprob)
        if masses:
            return dict(masses)
    return None


class JudgeScorer:
    """Step-level PRM phrasing: a running particle is judged on its latest step with the
    earlier steps as context; the round a particle finishes, its complete trajectory is
    judged with the outcome template instead."""

    name = "judge"

    def __init__(
        self,
        endpoint: str,
        model: str,
        fmt: StepFormat,
        concurrency: int = 48,
        timeout_s: float = 300.0,
        max_retries: int = 3,
    ):
        self.model = model
        self.fmt = fmt
        self.client = AsyncOpenAI(
            base_url=endpoint, api_key="EMPTY", timeout=timeout_s, max_retries=max_retries
        )
        self.semaphore = asyncio.Semaphore(concurrency)
        self.calls = 0
        self.failures = 0

    def prompt_for(self, record: MCQRecord, steps: list[str], finished: bool) -> str:
        if finished:
            return prompts.judge_outcome_prompt(record, self.fmt.trajectory(steps))
        if len(steps) == 1:
            return prompts.judge_step_prompt(record, "", steps[0])
        return prompts.judge_step_prompt(record, self.fmt.trajectory(steps[:-1]), steps[-1])

    async def _score_one(self, record: MCQRecord, prompt: str) -> StepScore:
        # Text first, then audio, as the judge was validated with.
        content = [{"type": "text", "text": prompt}, *prompts.judge_audio_parts(record.audio_paths)]
        async with self.semaphore:
            self.calls += 1
            try:
                completion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    temperature=1.0,
                    max_tokens=1,
                    logprobs=True,
                    top_logprobs=JUDGE_TOP_LOGPROBS,
                )
            except APIError:  # after the client's own retries; the caller applies the policy
                self.failures += 1
                return StepScore(score=None, failed=True)
        top = completion.choices[0].logprobs.content[0].top_logprobs
        return StepScore(score=yes_probability([(e.token, e.logprob) for e in top]))

    async def score(
        self,
        record: MCQRecord,
        base_messages: list[dict],
        particles: list[tuple[list[str], bool]],
    ) -> list[StepScore]:
        return list(
            await asyncio.gather(
                *(
                    self._score_one(record, self.prompt_for(record, steps, finished))
                    for steps, finished in particles
                )
            )
        )

    async def close(self) -> None:
        await self.client.close()


class ProbeScorer:
    """Self-probe: append "Answer:" to the trajectory (or cut at the model's own
    "Answer:") and read the greedy next-token distribution over option letters."""

    name = "probe"

    def __init__(self, generator: GeneratorClient, fmt: StepFormat, max_tokens: int = 3):
        self.generator = generator
        self.fmt = fmt
        self.max_tokens = max_tokens

    def prefix_for(self, steps: list[str]) -> str:
        text = self.fmt.trajectory(steps)
        cut = text.find(FINAL_MARKER)
        if cut >= 0:
            return text[: cut + len(FINAL_MARKER)]
        return text + ("" if text.endswith("\n") else "\n") + FINAL_MARKER

    async def _score_one(self, record, base_messages, steps) -> StepScore:
        positions = await self.generator.forced_answer_logprobs(
            base_messages, self.prefix_for(steps), self.max_tokens
        )
        masses = letter_masses(positions, LETTERS[: max(1, len(record.choices))])
        if not masses:
            return StepScore(score=None)
        return StepScore(score=max(masses.values()), letter_masses=masses)

    async def score(
        self,
        record: MCQRecord,
        base_messages: list[dict],
        particles: list[tuple[list[str], bool]],
    ) -> list[StepScore]:
        return list(
            await asyncio.gather(
                *(self._score_one(record, base_messages, steps) for steps, _ in particles)
            )
        )

    async def close(self) -> None:
        return None


class RandomScorer:
    """Control arm: every score is an independent Uniform(0, 1) draw that ignores the
    trajectory. Resampling then copies particles at random, so the gap between this and
    a real scorer is what the scorer's signal is worth.

    Draws are seeded per item and round, so a resumed item gets the same scores."""

    name = "random"

    def __init__(self, seed: int):
        self.seed = seed

    async def score(
        self,
        record: MCQRecord,
        base_messages: list[dict],
        particles: list[tuple[list[str], bool]],
    ) -> list[StepScore]:
        steps_so_far = sum(len(steps) for steps, _ in particles)
        rng = np.random.default_rng(
            [self.seed, zlib.crc32(record.unique_id.encode()), len(particles), steps_so_far]
        )
        return [StepScore(score=float(s)) for s in rng.uniform(size=len(particles))]

    async def close(self) -> None:
        return None
