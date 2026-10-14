"""Prompt construction for the generator and the judge.

The generator prompt is method 12 ("stepwise narrowing") of the previous
implementation, byte-identical (tests/test_parity.py). The judge prompts are the
templates of `babel-ship/judge_client.py`, also byte-identical.
"""

from __future__ import annotations

import os

from epf_spf.answers import LETTERS
from epf_spf.records import MCQRecord

GENERATOR_SYSTEM_PROMPT = "You are an expert audio analyst. Listen to the audio carefully."

STEPWISE_INSTRUCTION = (
    "Reason step by step toward the single best option, writing each "
    'step as "Step N: <reasoning>" on its own line. Each step should '
    "narrow down the options based on audio evidence -- don't just "
    "describe the audio in order. Use no more than 10 steps. "
    "Then end with 'Answer: <letter>'."
)

THINK_TRIGGER = "Please think and reason about the input {word} before you respond."

JUDGE_OUTCOME_TEMPLATE = """## ROLE
You are an expert evaluator judging answers to audio reasoning questions.

## TASK
Listen to the audio and read the question below. Determine whether the
candidate answer (including its reasoning) is CORRECT and well-supported
by the audio.

## Question
{question}

## Candidate Answer
{candidate}

## INSTRUCTION
Respond with exactly one word: "Yes" if the candidate answer is correct,
or "No" if it is incorrect. Do not provide any explanation.
"""

JUDGE_STEP_TEMPLATE = """## ROLE
You are an expert evaluator judging step-by-step reasoning on audio reasoning questions.

## TASK
Listen to the audio and read the question below. You are shown the candidate's
reasoning so far (context only) and its LATEST STEP. The reasoning is still in
progress — it may not state a final answer yet, and it must NOT be penalized
for being incomplete. Judge ONLY the latest step: is it correct, consistent
with what is actually audible in the audio, and does it advance toward the
correct option?

## Question
{question}

## Reasoning so far (context)
{context}

## Latest step (evaluate this)
{latest_step}

## INSTRUCTION
Respond with exactly one word: "Yes" if the latest step is correct and
advances toward the right answer, or "No" if it contains a mistake,
contradicts the audio, or leads toward a wrong answer. Do not provide any
explanation.
"""

FIRST_STEP_CONTEXT = "(none — this is the first step)"


def format_choices(choices: list[str]) -> str:
    return "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))


def generator_audio_parts(audio_paths: list[str]) -> list[dict]:
    # The generator path uses abspath (not realpath), as the previous implementation did.
    return [
        {"type": "audio_url", "audio_url": {"url": "file://" + os.path.abspath(p)}}
        for p in audio_paths
    ]


def judge_audio_parts(audio_paths: list[str]) -> list[dict]:
    # vLLM resolves symlinks before its allowed-media-path check, so the judge
    # path must be the realpath or requests are rejected on a healthy server.
    return [
        {"type": "audio_url", "audio_url": {"url": "file://" + os.path.realpath(p)}}
        for p in audio_paths
    ]


def think_trigger_word(category: str | None) -> str:
    category = (category or "").strip().lower()
    return category if category in ("music", "speech", "sound") else "audio"


def generator_messages(record: MCQRecord, think_trigger: bool = False) -> list[dict]:
    """System and user turns of the method-12 prompt; audio parts precede the text."""
    text = (
        f"Question: {record.question}\nOptions:\n{format_choices(record.choices)}\n"
        + STEPWISE_INSTRUCTION
    )
    if think_trigger:
        text = (
            text.rstrip() + "\n\n" + THINK_TRIGGER.format(word=think_trigger_word(record.category))
        )
    return [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [*generator_audio_parts(record.audio_paths), {"type": "text", "text": text}],
        },
    ]


def judge_question(record: MCQRecord) -> str:
    return f"Question: {record.question}\n\nOptions:\n{format_choices(record.choices)}"


def judge_step_prompt(record: MCQRecord, context: str, latest_step: str) -> str:
    return JUDGE_STEP_TEMPLATE.format(
        question=judge_question(record),
        context=context or FIRST_STEP_CONTEXT,
        latest_step=latest_step,
    )


def judge_outcome_prompt(record: MCQRecord, trajectory: str) -> str:
    return JUDGE_OUTCOME_TEMPLATE.format(question=judge_question(record), candidate=trajectory)
